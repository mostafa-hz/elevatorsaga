const createRandomAgent = function(options) {
    const { floorCount, elevatorCount } = options;
    const actionSize = floorCount;

    function getRandomAction() {
        return Math.floor(Math.random() * actionSize);
    }

    return {
        step: function(world) {
            const observation = {};
            const action = getRandomAction();
            return { observation, action };
        },
    }
};

const createShabbatAgent = function(options) {
    const { floorCount, elevatorCount } = options;
    const actionSize = floorCount;
    let lastIndex = 0;

    function getNextAction() {
        lastIndex = (lastIndex + 1) % actionSize;
        return lastIndex;
    }

    return {
        step: function(world) {
            const observation = { lastIndex };
            const action = getNextAction();
            return getNextAction({ observation, action });
        },
    }
};

const createDeepAgent = async function(options, modelFiles) {
    const { floorCount, elevatorCount } = options;
    const actionSize = floorCount;

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
            envState[`f${i}_PU`] = floor.buttonStates.up ? world.elapsedTime - floor.buttonStates.upElapsedTime : -5;
            envState[`f${i}_PD`] = floor.buttonStates.down ? world.elapsedTime - floor.buttonStates.downElapsedTime : -5;
        }

        return envState;
    }

    function generateNetInput(state) {
        return Object.values(state);
    }

    function getBestAction(state) {
        const input = generateNetInput(state);
        const expectedRewards = model.predict(tf.tensor([input])).dataSync();

        let maxIndex = 0;
        expectedRewards.forEach((reward, i) => {
            if(reward > expectedRewards[maxIndex]) {
                maxIndex = i;
            }
        });
        return maxIndex;
    }

    function getRandomAction() {
        return Math.floor(Math.random() * actionSize);
    }

    async function loadModel() {
        const modelFile = Object.values(modelFiles).find(it => it.name.endsWith('.model.json'));
        const weightsFile = Object.values(modelFiles).find(it => it.name.endsWith('.model.weights.bin'));
        return tf.loadLayersModel(tf.io.browserFiles([modelFile, weightsFile]));
    }

    function buildModel() {
        const inputSize = (floorCount * 2) + (elevatorCount * 5) + (floorCount * elevatorCount);
        return tf.sequential({
            layers: [
                tf.layers.dense({ inputShape: [inputSize], units: inputSize * 2 }),
                tf.layers.leakyReLU(),
                tf.layers.dense({ units: inputSize }),
                tf.layers.leakyReLU(),
                tf.layers.dense({ units: actionSize }),
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

    let trainModel;
    let trainCount = 0;

    return {
        step: function(world, explore = false) {
            const observation = observe(world);
            const action = explore ? getRandomAction(world) : getBestAction(observation);

            return { observation, action };
        },

        train: async function(memory) {
            if(trainModel == null) {
                trainModel = modelFiles?.length === 2 ? await loadModel() : buildModel();
                trainModel.setWeights(model.getWeights());
            }

            const discount = 0.9;
            const {
                observations,
                actions,
                rewards,
            } = memory;

            const netInputs = observations.map(generateNetInput);
            const expectedTargets = trainModel.predict(tf.tensor(netInputs)).dataSync();
            const targets = rewards.map((reward, i) => {
                const nextTarget = expectedTargets.slice((i + 1) * actionSize, (i + 2) * actionSize);
                const nextReward = nextTarget.length ? Math.max(...nextTarget) : 0;
                nextTarget[actions[i]] = reward + nextReward * discount;
                return nextTarget;
            });

            await model.fit(tf.tensor(netInputs), tf.tensor(targets));

            if(trainCount % 10 === 0){
                trainModel.setWeights(model.getWeights());
            }
            trainCount++;
        },

        saveModel: async function(name) {
            await model.save(`downloads://${name}`);
        },
    }
};
