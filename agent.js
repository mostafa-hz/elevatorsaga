const createAgent = function (options) {
    const {floorCount, elevatorCount} = options;

    // TODO fix for multiple elevators
    const possibleActions = [];
    for (let i = 0; i < floorCount ** elevatorCount; i++) {
        let action = [];
        for (let j = 0; j < elevatorCount; j++) {
            action.push(i)
        }
        possibleActions.push(action);
    }

    function observe(world) {
        const envState = {};
        const elevators = world.elevatorInterfaces;
        const floors = world.floors;
        for (let i = 0; i < elevators.length; i++) {
            const elevator = elevators[i];
            // envState[`eMaxPassengerCount${i}`] = elevator.maxPassengerCount();
            envState[`eCurrentFloor${i}`] = elevator.currentFloor();
            envState[`eLoadFactor${i}`] = elevator.loadFactor();
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
            envState[`eDestinationDirection${i}`] = direction;
            const pressedFloors = new Array(floors.length);
            for (let j of elevator.getPressedFloors()) {
                pressedFloors[j] = true
            }
            for (let j = 0; j < floors.length; j++) {
                envState[`ePressedFloor${i}${j}`] = Number(pressedFloors[j] === true);
            }
        }
        for (let i = 0; i < floors.length; i++) {
            const floor = floors[i];
            envState[`fUpPressed${i}`] = Number(floor.buttonStates.up === 'activated');
            envState[`fDownPressed${i}`] = Number(floor.buttonStates.down === 'activated');
        }

        return envState;
    }

    function calculateReward(world, oldWorld, observation, oldObservation) {
        // TODO Fix For Multiple elevators
        const {
            elapsedTime,
            transportedPerSec,
            maxWaitTime,
            avgWaitTime,
            moveCount,
            transportedCounter,
        } = world;

        const {
            elapsedTime: elapsedTimeOld,
            transportedPerSec: transportedPerSecOld,
            maxWaitTime: maxWaitTimeOld,
            avgWaitTime: avgWaitTimeOld,
            moveCount: moveCountOld,
            transportedCounter: transportedCounterOld,
        } = oldWorld;

        const timeDelta = elapsedTime - elapsedTimeOld;

        let buttonsPressed = 0;
        for (let i = 0; i < floorCount; i++) {
            buttonsPressed += observation[`fUpPressed${i}`];
            buttonsPressed += observation[`fDownPressed${i}`];
        }
        const floorsPressed = buttonsPressed * timeDelta;

        let loadAvg = 0;
        let oldLoadAvg = 0;
        let notMoving = 0;
        let changeDirection = 0;
        for (let i = 0; i < elevatorCount; i++) {
            loadAvg += observation[`eLoadFactor${i}`];
            oldLoadAvg += oldObservation[`eLoadFactor${i}`];
            const oldDirection = oldObservation[`eDestinationDirection${i}`];
            const direction = observation[`eDestinationDirection${i}`];
            if (oldDirection !== direction) {
                changeDirection = 1
            }
            if (direction === 0) {
                for (let j = 0; j < floorCount; j++) {
                    if (observation[`ePressedFloor${i}${j}`] === 1) {
                        notMoving = 1;
                        break;
                    }
                }
            }

        }
        loadAvg /= elevatorCount;
        oldLoadAvg /= elevatorCount;

        const transported = Math.abs(loadAvg - oldLoadAvg);

        let reward = 0;
        reward += (transported * 50);
        reward += ((changeDirection * loadAvg) * -20);
        reward += floorsPressed * -1;
        reward += notMoving * -50;

        return reward;
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

    function getBestAction(state) {
        let maxRewardIndex = 0;
        let maxReward = -Infinity;
        possibleActions.forEach((action, i) => {
            const input = generateNetInput(state, action);
            const expectedReward = model.predict(tf.tensor([input])).dataSync()[0];
            if (expectedReward > maxReward) {
                maxReward = expectedReward;
                maxRewardIndex = i;
            }
        });
        return possibleActions[maxRewardIndex]
    }

    let preRandomIndex = 0;

    function getRandomAction() {
        const randomIndex = Math.random() > 0.5 ? Math.floor(Math.random() * possibleActions.length) : preRandomIndex;
        preRandomIndex = randomIndex;
        return possibleActions[randomIndex]
    }


    // buildModel
    const statesSize = (floorCount * 2) + (elevatorCount * 3) + (floorCount * elevatorCount);
    const actionSize = floorCount * elevatorCount;
    const inputSize = statesSize + actionSize;
    const model = tf.sequential({
        layers: [
            tf.layers.dense({inputShape: [inputSize], units: inputSize, activation: 'relu'}),
            tf.layers.dense({units: floorCount * elevatorCount * 3, activation: 'relu'}),
            tf.layers.dense({units: elevatorCount, activation: 'relu'}),
            tf.layers.dense({units: 1}),
        ]
    });
    model.compile({
        loss: tf.losses.meanSquaredError,
        optimizer: tf.train.adam(0.05),
        metrics: ['accuracy'],
    });

    return {
        play: async function (world, exploreRate = 0) {
            const memory = {
                observations: [],
                actions: [],
                rewards: [],
            };

            let cb = undefined;

            let oldWorld = {...world};
            world.on('stats_changed', function () {
                if (cb == null) return;
                const observation = observe(world);
                const oldObservation = memory.observations[memory.observations.length - 1] || {};
                if (
                    !world.challengeEnded &&
                    world.elapsedTime - 0.5 < oldWorld.elapsedTime && // prevent stuck in an state for exploration
                    String(Object.values(observation)) === String(Object.values(oldObservation))
                ) return;
                world.off('stats_changed', this);
                const reward = calculateReward(world, oldWorld, observation, oldObservation);
                cb(reward);
                oldWorld = {...world};
                cb = undefined;
            });

            async function takeAction(world, action) {
                const elevators = world.elevatorInterfaces;
                elevators.forEach((elevator, i) => elevator.goToFloor(action[i], true));

                return new Promise((resolve => {
                    cb = resolve
                }));
            }

            while (!world.challengeEnded) {
                const observation = observe(world);
                memory.observations.push(observation);

                const action = Math.random() > exploreRate ? getBestAction(observation) : getRandomAction();
                memory.actions.push(action);

                const reward = await takeAction(world, action);
                memory.rewards.push(reward);
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
