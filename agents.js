const createRandomAgent = function() {
    function getRandomAction(world) {
        const randomIndex = Math.floor(Math.random() * world.possibleActions.length);
        return world.possibleActions[randomIndex]
    }

    return {
        play: async function(world) {
            const memory = {
                observations: [],
                actions: [],
                rewards: [],
            };

            while(!world.challengeEnded) {
                const action = getRandomAction(world);
                const { end } = await world.takeAction(world, action);
                if(end) break;
            }

            return memory;
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
        play: async function(world) {
            while(!world.challengeEnded) {
                const action = getNextAction(world);
                const { end } = await world.takeAction(world, action);
                if(end) break;
            }
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
            envState[`f${i}_PU`] = Number(floor.buttonStates.up === 'activated');
            envState[`f${i}_PD`] = Number(floor.buttonStates.down === 'activated');
        }

        return envState;
    }

    function generateNetInput(state, action) {
        const stateInput = Object.values(state);
        let actionInput = [];
        for(const a of action) {
            for(let f = 0; f < floorCount; f++) {
                actionInput.push(f === a ? 1 : 0)
            }
        }
        return [...stateInput, ...actionInput]
    }

    function getBestAction(world, state) {
        const inputs = world.possibleActions.map(action => generateNetInput(state, action));
        const expectedRewards = model.predict(tf.tensor(inputs)).dataSync();

        let maxIndex = 0;
        expectedRewards.forEach((reward, i) => {
            if(reward > expectedRewards[expectedRewards]) {
                maxIndex = i;
            }
        });
        return world.possibleActions[maxIndex]
    }

    function getRandomAction(world) {
        const randomIndex = Math.floor(Math.random() * world.possibleActions.length);
        return world.possibleActions[randomIndex]
    }

    async function loadModel() {
        const modelFile = Object.values(modelFiles).find(it => it.name.endsWith('.model.json'));
        const weightsFile =  Object.values(modelFiles).find(it => it.name.endsWith('.model.weights.bin'));
        return tf.loadLayersModel(tf.io.browserFiles([modelFile, weightsFile]));
    }

    function buildModel() {
        const statesSize = (floorCount * 2) + (elevatorCount * 3) + (floorCount * elevatorCount);
        const actionSize = floorCount * elevatorCount;
        const inputSize = statesSize + actionSize;
        return tf.sequential({
            layers: [
                tf.layers.dense({ inputShape: [inputSize], units: inputSize }),
                tf.layers.leakyReLU(),
                tf.layers.dense({ units: 27 }),
                tf.layers.leakyReLU(),
                tf.layers.dense({ units: 9 }),
                tf.layers.leakyReLU(),
                tf.layers.dense({ units: 1 }),
            ]
        });
    }

    const model = modelFiles?.length === 2 ? await loadModel() : buildModel();
    model.compile({
        loss: tf.losses.meanSquaredError,
        optimizer: tf.train.adam(0.3),
        metrics: ['accuracy'],
    });

    return {
        play: async function(world, exploreRate = 0) {
            const memory = {
                observations: [],
                actions: [],
                rewards: [],
            };

            while(!world.challengeEnded) {
                const observation = observe(world);
                const explore = Math.random() < exploreRate;
                const action = explore ? getRandomAction(world) : getBestAction(world, observation);
                const { reward, end } = await world.takeAction(world, action);

                if(end) break;

                memory.observations.push(observation);
                memory.actions.push(action);
                memory.rewards.push(reward);
            }

            return memory;
        },

        train: async function(memory) {
            const {
                observations,
                actions,
                rewards,
            } = memory;

            const netInputs = observations.map((state, i) => generateNetInput(observations[i], actions[i]));
            await model.fit(tf.tensor(netInputs), tf.tensor(rewards));
        },

        saveModel: async function(name) {
            await model.save(`downloads://${name}`);
        },
    }
};
