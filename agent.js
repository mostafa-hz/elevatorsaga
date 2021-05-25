const createAgent = function (options) {
    const {floorCount, elevatorCount} = options;

    function observe(world) {
        const envState = {};
        const elevators = world.elevatorInterfaces;
        const floors = world.floors;
        for (let i = 0; i < elevators.length; i++) {
            const elevator = elevators[i];
            // envState[`eMaxPassengerCount${i}`] = elevator.maxPassengerCount();
            envState[`e${i}_CF`] = elevator.currentFloor();
            envState[`e${i}_LF`] = elevator.loadFactor();
            let direction;
            switch (elevator.destinationDirection()) {
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
            for (let j of elevator.getPressedFloors()) {
                pressedFloors[j] = true
            }
            for (let j = 0; j < floors.length; j++) {
                envState[`e${i}_PF${j}`] = Number(pressedFloors[j] === true);
            }
        }
        for (let i = 0; i < floors.length; i++) {
            const floor = floors[i];
            envState[`f${i}_PU`] = Number(floor.buttonStates.up === 'activated');
            envState[`f${i}_PD`] = Number(floor.buttonStates.down === 'activated');
        }

        return envState;
    }

    function generateNetInput(state, action) {
        const stateInput = Object.values(state);
        let actionInput = [];
        for (const a of action) {
            for (let f = 0; f < floorCount; f++) {
                actionInput.push(f === a ? 1 : 0)
            }
        }
        return [...stateInput, ...actionInput]
    }

    function getBestAction(world, state) {
        let maxRewardIndex = 0;
        let maxReward = -Infinity;
        world.possibleActions.forEach((action, i) => {
            const input = generateNetInput(state, action);
            const expectedReward = model.predict(tf.tensor([input])).dataSync()[0];
            if (expectedReward > maxReward) {
                maxReward = expectedReward;
                maxRewardIndex = i;
            }
        });
        return world.possibleActions[maxRewardIndex]
    }

    function getRandomAction(world) {
        const randomIndex = Math.floor(Math.random() * world.possibleActions.length);
        return world.possibleActions[randomIndex]
    }

    // buildModel
    const statesSize = (floorCount * 2) + (elevatorCount * 3) + (floorCount * elevatorCount);
    const actionSize = floorCount * elevatorCount;
    const inputSize = statesSize + actionSize;
    const model = tf.sequential({
        layers: [
            tf.layers.dense({inputShape: [inputSize], units: inputSize, activation: 'linear'}),
            tf.layers.dense({units: 27, activation: 'linear'}),
            tf.layers.dense({units: 9, activation: 'linear'}),
            tf.layers.dense({units: 3, activation: 'linear'}),
            tf.layers.dense({units: 1}),
        ]
    });
    model.compile({
        loss: tf.losses.meanSquaredError,
        optimizer: tf.train.adam(0.3),
        metrics: ['accuracy'],
    });

    return {
        play: async function (world, exploreRate = 0) {
            const memory = {
                observations: [],
                actions: [],
                rewards: [],
            };

            while (!world.challengeEnded) {
                const observation = observe(world);
                const action = Math.random() > exploreRate ? getBestAction(world, observation) : getRandomAction(world);
                const {reward, end} = await world.takeAction(world, action);

                if (end) break;

                memory.observations.push(observation);
                memory.actions.push(action);
                memory.rewards.push(reward);

                console.log(action[0], '->', reward);
            }

            console.log('play done');

            return memory;
        },

        train: async function (memory) {
            const {
                observations,
                actions,
                rewards,
            } = memory;

            const accReward = rewards.reduce((pre, curr) => pre + curr, 0);

            console.log('r:', accReward, 't:', world.avgWaitTime, 'c:', world.moveCount, 'tc:', world.transportedCounter,);

            const netInputs = observations.map((state, i) => generateNetInput(observations[i], actions[i]));
            await model.fit(tf.tensor(netInputs), tf.tensor(rewards));
        },

        saveModel: async function (name) {
            // TODO
        },

        loadModel: async function (name) {
            // TODO
        }
    }
};
