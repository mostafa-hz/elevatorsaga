const createRandomAgent = function() {
    function getRandomAction(world) {
        const randomIndex = Math.floor(Math.random() * world.possibleActions.length);
        return world.possibleActions[randomIndex]
    }

    return {
        step: function(world) {
            const observation = {};
            const action = getRandomAction(world);
            return { observation, action };
        },
    }
};

const createShabbatAgent = function() {
    let lastIndex = 0;

    function getNextAction(world) {
        lastIndex = (lastIndex + 1) % world.possibleActions.length;
        return world.possibleActions[lastIndex];
    }

    return {
        step: function(world) {
            const observation = { lastIndex };
            const action = getNextAction(world);
            return getNextAction({ observation, action });
        },
    }
};

const createDeepAgent = async function(options, modelFiles) {
    const { floorCount, elevatorCount } = options;

    function observe(world) {
        const envState = {};
        const elevators = world.elevatorInterfaces;
        const floors = world.floors;
        for(let i = 0; i < elevators.length; i++) {
            const elevator = elevators[i];
            // envState[`eMaxPassengerCount${i}`] = elevator.maxPassengerCount();
            envState[`e${i}_CF`] = elevator.currentFloor();
            envState[`e${i}_LF`] = elevator.loadFactor();
            envState[`e${i}_IU`] = elevator.goingDownIndicator();
            envState[`e${i}_ID`] = elevator.goingDownIndicator();
            let direction;
            switch(elevator.destinationDirection()) {
                case 'up':
                    direction = 1;
                    break;
                case 'down':
                    direction = -1;
                    break;
                default:
                    direction = 0;
                    break;
            }
            envState[`e${i}_DD`] = direction;
            const pressedFloors = new Array(floors.length);
            for(let j of elevator.getPressedFloors()) {
                pressedFloors[j] = true
            }
            for(let j = 0; j < floors.length; j++) {
                envState[`e${i}_PF${j}`] = Number(pressedFloors[j] === true);
            }
        }
        for(let i = 0; i < floors.length; i++) {
            const floor = floors[i];
            envState[`f${i}_PU`] = floor.buttonStates.up ? world.elapsedTime - floor.buttonStates.upElapsedTime : 0;
            envState[`f${i}_PD`] = floor.buttonStates.down ? world.elapsedTime - floor.buttonStates.downElapsedTime : 0;
        }

        return envState;
    }

    function generateNetInput(state, action) {
        const stateInput = Object.values(state);
        let actionInput = [];
        for(const a of action) {
            actionInput.push(a.indicator);
            for(let f = 0; f < floorCount; f++) {
                actionInput.push(f === a.floor ? 1 : 0)
            }
        }
        return [...stateInput, ...actionInput]
    }

    function getBestAction(possibleActions, state) {
        const inputs = possibleActions.map(action => generateNetInput(state, action));
        const expectedRewards = model.predict(tf.tensor(inputs)).dataSync();

        let maxIndex = 0;
        expectedRewards.forEach((reward, i) => {
            if(reward > expectedRewards[maxIndex]) {
                maxIndex = i;
            }
        });
        return possibleActions[maxIndex]
    }

    function getRandomAction(world) {
        const randomIndex = Math.floor(Math.random() * world.possibleActions.length);
        return world.possibleActions[randomIndex]
    }

    async function loadModel() {
        const modelFile = Object.values(modelFiles).find(it => it.name.endsWith('.model.json'));
        const weightsFile = Object.values(modelFiles).find(it => it.name.endsWith('.model.weights.bin'));
        return tf.loadLayersModel(tf.io.browserFiles([modelFile, weightsFile]));
    }

    function buildModel() {
        const statesSize = (floorCount * 2) + (elevatorCount * 5) + (floorCount * elevatorCount);
        const actionSize = (floorCount * elevatorCount) + elevatorCount;
        const inputSize = statesSize + actionSize;
        return tf.sequential({
            layers: [
                tf.layers.dense({ inputShape: [inputSize], units: inputSize * 2 }),
                tf.layers.leakyReLU(),
                tf.layers.dense({ units: inputSize }),
                tf.layers.leakyReLU(),
                tf.layers.dense({ units: 1 }),
            ]
        });
    }

    const model = modelFiles?.length === 2 ? await loadModel() : buildModel();
    model.compile({
        loss: tf.losses.meanSquaredError,
        optimizer: tf.train.adam(0.01),
        metrics: ['mse'],
    });
    model.summary();

    return {
        step: function(world, explore = false) {
            const { possibleActions } = world;
            const observation = observe(world);
            const action = explore ? getRandomAction(world) : getBestAction(possibleActions, observation);

            return { observation, action };
        },

        train: async function(memory) {
            const discount = 0.9;
            const {
                possibleActions,
                observations,
                actions,
                rewards,
            } = memory;

            const nextRewards = observations.slice(1).map(state => {
                const inputs = possibleActions.map(action => generateNetInput(state, action));
                const expectedRewards = model.predict(tf.tensor(inputs)).dataSync().values();
                return Math.max(...expectedRewards);
            });
            let targets = rewards.map((reward, i) => reward + (nextRewards[i] ?? 0) * discount);

            const netInputs = observations.map((state, i) => generateNetInput(observations[i], actions[i]));
            await model.fit(tf.tensor(netInputs), tf.tensor(targets));
        },

        saveModel: async function(name) {
            await model.save(`downloads://${name}`);
        },
    }
};
