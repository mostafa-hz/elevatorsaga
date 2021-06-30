var createEditor = function() {
    var lsKey = "elevatorCrushCode_v5";

    var cm = CodeMirror.fromTextArea(document.getElementById("code"), {
        lineNumbers: true,
        indentUnit: 4,
        indentWithTabs: false,
        theme: "solarized light",
        mode: "javascript",
        autoCloseBrackets: true,
        extraKeys: {
            // the following Tab key mapping is from http://codemirror.net/doc/manual.html#keymaps
            Tab: function(cm) {
                var spaces = new Array(cm.getOption("indentUnit") + 1).join(" ");
                cm.replaceSelection(spaces);
            }
        }
    });

    // reindent on paste (adapted from https://github.com/ahuth/brackets-paste-and-indent/blob/master/main.js)
    cm.on("change", function(codeMirror, change) {
        if(change.origin !== "paste") {
            return;
        }

        var lineFrom = change.from.line;
        var lineTo = change.from.line + change.text.length;

        function reindentLines(codeMirror, lineFrom, lineTo) {
            codeMirror.operation(function() {
                codeMirror.eachLine(lineFrom, lineTo, function(lineHandle) {
                    codeMirror.indentLine(lineHandle.lineNo(), "smart");
                });
            });
        }

        reindentLines(codeMirror, lineFrom, lineTo);
    });

    var reset = function() {
        cm.setValue($("#default-elev-implementation").text().trim());
    };
    var saveCode = function() {
        localStorage.setItem(lsKey, cm.getValue());
        $("#save_message").text("Code saved " + new Date().toTimeString());
        returnObj.trigger("change");
    };

    var existingCode = localStorage.getItem(lsKey);
    if(existingCode) {
        cm.setValue(existingCode);
    } else {
        reset();
    }

    var returnObj = riot.observable({});
    var autoSaver = _.debounce(saveCode, 1000);
    cm.on("change", function() {
        autoSaver();
    });

    returnObj.getCodeObj = function() {
        var code = cm.getValue();
        var obj;
        try {
            obj = getCodeObjFromCode(code);
            returnObj.trigger("code_success");
        } catch(e) {
            returnObj.trigger("usercode_error", e);
            return null;
        }
        return obj;
    };
    returnObj.setCode = function(code) {
        cm.setValue(code);
    };
    returnObj.getCode = function() {
        return cm.getValue();
    }
    returnObj.setDevTestCode = function() {
        cm.setValue($("#devtest-elev-implementation").text().trim());
    }

    $("#button_apply").click(function() {
        returnObj.trigger("apply_code");
    });

    $("#button_export_model").click(function() {
        returnObj.trigger("export_model");
    });

    function setImportedFileText() {
        const fileInput = $("#file_import_model")[0];

        let importedFileNames = '';
        fileInput?.files?.forEach(it => {
            importedFileNames += '<br>' + it.name
        });
        if(importedFileNames.trim().length === 0) {
            importedFileNames = 'none';
        }
        $('#text_imported_file')[0].innerHTML = `imported files: ${importedFileNames}`;
    }

    $('#file_import_model').change(setImportedFileText);
    $('#button_reset').click(setImportedFileText);
    setImportedFileText();

    return returnObj;
};


var createParamsUrl = function(current, overrides) {
    return "#" + _.map(_.merge(current, overrides), function(val, key) {
        return key + "=" + val;
    }).join(",");
};


$(function() {
    var tsKey = "elevatorTimeScale";
    var editor = createEditor();

    var params = {};

    var $world = $(".innerworld");
    var $stats = $(".statscontainer");
    var $feedback = $(".feedbackcontainer");
    var $challenge = $(".challenge");

    var floorTempl = document.getElementById("floor-template").innerHTML.trim();
    var elevatorTempl = document.getElementById("elevator-template").innerHTML.trim();
    var elevatorButtonTempl = document.getElementById("elevatorbutton-template").innerHTML.trim();
    var userTempl = document.getElementById("user-template").innerHTML.trim();
    var challengeTempl = document.getElementById("challenge-template").innerHTML.trim();
    var feedbackTempl = document.getElementById("feedback-template").innerHTML.trim();

    var app = riot.observable({});
    app.worldController = createWorldController(1.0 / 60.0);
    app.worldController.on("usercode_error", function(e) {
        console.log("World raised code error", e);
        editor.trigger("usercode_error", e);
    });

    console.log(app.worldController);
    app.worldCreator = createWorldCreator();
    app.world = undefined;
    app.agent = undefined;

    app.currentChallengeIndex = 0;

    app.startStopOrRestart = function() {
        if(app.world.challengeEnded) {
            app.startChallenge(app.currentChallengeIndex);
        } else {
            app.worldController.setPaused(!app.worldController.isPaused);
        }
    };

    app.startChallenge = function(challengeIndex, autoStart) {
        if(typeof app.world !== "undefined") {
            app.world.unWind();
            // TODO: Investigate if memory leaks happen here
        }
        app.currentChallengeIndex = challengeIndex;
        app.world = app.worldCreator.createWorld(challenges[challengeIndex].options);
        window.world = app.world;

        clearAll([$world, $feedback]);
        presentStats($stats, app.world);
        presentChallenge($challenge, challenges[challengeIndex], app, app.world, app.worldController, challengeIndex + 1, challengeTempl);
        presentWorld($world, app.world, floorTempl, elevatorTempl, elevatorButtonTempl, userTempl);

        app.worldController.on("timescale_changed", function() {
            localStorage.setItem(tsKey, app.worldController.timeScale);
            presentChallenge($challenge, challenges[challengeIndex], app, app.world, app.worldController, challengeIndex + 1, challengeTempl);
        });

        app.world.on("stats_changed", function() {
            var challengeStatus = challenges[challengeIndex].condition.evaluate(app.world);
            if(challengeStatus !== null) {
                app.world.endChallenge();
                app.worldController.setPaused(true);
                if(challengeStatus) {
                    presentFeedback($feedback, feedbackTempl, app.world, "Success!", "Challenge completed", createParamsUrl(params, { challenge: (challengeIndex + 2) }));
                } else {
                    presentFeedback($feedback, feedbackTempl, app.world, "Challenge failed", "Maybe your program needs an improvement?", "");
                }
            }
            const elevator = app.world.elevatorInterfaces[0];
            const idleElevator = elevator.destinationQueue.length === 0;
            if(idleElevator && !elevator.isBusy()) {
                const { action } = app.agent.step(app.world);
                app.world.takeAction(action);
            }
        });

        var codeObj = editor.getCodeObj();
        console.log("Starting...");
        app.worldController.start(app.world, codeObj, window.requestAnimationFrame, autoStart);
    };

    editor.on("apply_code", async function() {
        let agentSelector = $("#selector_agents")[0];
        const agentType = agentSelector.value;
        switch(agentType) {
            case 'random':
                app.agent = createRandomAgent(challenges[app.currentChallengeIndex].options);
                break;
            case 'shabbat':
                app.agent = createShabbatAgent(challenges[app.currentChallengeIndex].options);
                break;
            case 'deep':
                const fileInput = $("#file_import_model")[0];
                app.agent = await createDeepAgent(challenges[app.currentChallengeIndex].options, fileInput.files);
                break;
        }
    });
    editor.on("export_model", async function() {
        if(app.agent?.saveModel != null) {
            const { floorCount, elevatorCount } = challenges[app.currentChallengeIndex].options;
            await app.agent.saveModel(`agent-F${floorCount}E${elevatorCount}-${new Date().getTime()}.model`);
        }
    });
    editor.trigger("change");

    riot.route(function(path) {
        params = _.reduce(path.split(","), function(result, p) {
            var match = p.match(/(\w+)=(\w+$)/);
            if(match) {
                result[match[1]] = match[2];
            }
            return result;
        }, {});
        var requestedChallenge = 0;
        var autoStart = false;
        var timeScale = parseFloat(localStorage.getItem(tsKey)) || 2.0;
        _.each(params, function(val, key) {
            if(key === "challenge") {
                requestedChallenge = _.parseInt(val) - 1;
                if(requestedChallenge < 0 || requestedChallenge >= challenges.length) {
                    console.log("Invalid challenge index", requestedChallenge);
                    console.log("Defaulting to first challenge");
                    requestedChallenge = 0;
                }
            } else if(key === "autostart") {
                autoStart = val === "false" ? false : true;
            } else if(key === "timescale") {
                timeScale = parseFloat(val);
            } else if(key === "devtest") {
                editor.setDevTestCode();
            } else if(key === "fullscreen") {
                makeDemoFullscreen();
            }
        });
        app.worldController.setTimeScale(timeScale);
        app.startChallenge(requestedChallenge, autoStart);
    });

    let train = false;
    let trainEpisode = 0;

    $('#input_train').click(async function() {
        train = !train;
        $('#input_train')[0].innerHTML = train ? 'stop' : 'start';
        if(train) trainModel();
    });

    async function trainModel() {
        const codeObj = {
            init() {
            },
            update() {
            }
        };
        const challenge = challenges[app.currentChallengeIndex];
        let exploreRate = 1;
        while(train) {
            const { memory, result } = runEpisode(challenge, codeObj, 1000.0 / 60.0, 12000, exploreRate);
            console.log(result);
            await app.agent.train(memory);
            if(exploreRate > 0.25) exploreRate -= 0.001;
            trainEpisode++;
            $('#p_train_count')[0].innerHTML = `${trainEpisode} episodes`;
        }
    }

    function runEpisode(challenge, codeObj, stepSize, stepsToSimulate, exploreRate) {
        const controller = createWorldController(stepSize);

        const worldCreator = createWorldCreator();
        const world = worldCreator.createWorld(challenge.options);
        const frameRequester = createFrameRequester(stepSize);

        controller.start(world, codeObj, frameRequester.register, true);

        const memory = {
            possibleActions: world.possibleActions,
            observations: [],
            actions: [],
            rewards: [],
        };

        let accReward = 0;
        for(let stepCount = 0; stepCount < stepsToSimulate && !controller.isPaused; stepCount++) {
            frameRequester.trigger();
            const elevator = world.elevatorInterfaces[0];
            const idleElevator = elevator.destinationQueue.length === 0;
            if(!idleElevator || elevator.isBusy()) continue;
            if(stepCount > 0) {
                const reward = world.calculateReward();
                memory.rewards.push(reward);
                accReward += reward;
            }

            const explore = exploreRate > Math.random();
            const { observation, action } = app.agent.step(world, explore);
            world.takeAction(action);
            memory.observations.push(observation);
            memory.actions.push(action);
        }
        world.endChallenge();

        // remove last step with no reward
        memory.observations = memory.observations.slice(0, memory.rewards.length);
        memory.actions = memory.actions.slice(0, memory.rewards.length);

        const result = {
            exploreRate,
            transportedPerSec: world.transportedPerSec,
            avgWaitTime: world.avgWaitTime,
            maxWaitTime: world.maxWaitTime,
            transportedCount: world.transportedCounter,
            accReward,
        };

        return { memory, result };
    }
});
