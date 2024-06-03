var canvasWidth = 800;
var canvasHeight = 400;
var player;
var background;
var newGame = true;

var bullet, bulletD = false, ship;
var bullet2, bulletD2 = false, ship2;
var bullet3, bulletD3 = false, ship3;
var jump, left, right, direction = 1;
var menu;

var bulletSpeed;
var bulletSpeed2 = 203;
var bulletSpeed3 = 160; 
var bulletDistance;
var bulletDistance2;
var bulletDistance2x; 
var bulletDistance3;
var bulletDistance3x;
var airStatus;
var groundStatus;
var progress;
var still;
var nnNetwork2, nnTraining2, nnOutput2, trainingData2 = [];
var nnNetwork, nnTraining, nnOutput, trainingData = [];
var nnNetwork3, nnTraining3, nnOutput3, trainingData3 = [];
var autoMode = false, trainingComplete = false;

var game = new Phaser.Game(canvasWidth, canvasHeight, Phaser.CANVAS, '', {
    preload: preload,
    create: create,
    update: update,
    render: render
});

function preload() {
    game.load.image('background', 'assets/game/background1.jpg');
    game.load.spritesheet('character', 'assets/sprites/altair.png', 32, 48);
    game.load.image('ship', 'assets/game/ufo.png');
    game.load.image('bullet', 'assets/sprites/purple_ball.png');
    game.load.image('menu', 'assets/game/menu.png');
}

function create() {
    game.physics.startSystem(Phaser.Physics.ARCADE);
    game.physics.arcade.gravity.y = 800;
    game.time.desiredFps = 30;

    background = game.add.tileSprite(0, 0, canvasWidth, canvasHeight, 'background');
    ship = game.add.sprite(canvasWidth - 100, canvasHeight - 70, 'ship');
    ship2 = game.add.sprite(20, canvasHeight - 400, 'ship');
    ship3 = game.add.sprite(canvasWidth - 100, 0, 'ship'); 
    bullet = game.add.sprite(canvasWidth - 100, canvasHeight, 'bullet');
    bullet2 = game.add.sprite(55, canvasHeight - 350, 'bullet');
    bullet3 = game.add.sprite(canvasWidth - 100, 70, 'bullet'); 
    player = game.add.sprite(50, canvasHeight - 48, 'character'); 

    game.physics.enable(player);
    player.body.collideWorldBounds = true;
    var run = player.animations.add('run', [8, 9, 10, 11]);
    player.animations.play('run', 10, true);

    game.physics.enable(bullet);
    bullet.body.collideWorldBounds = true;

    game.physics.enable(bullet2);
    bullet2.body.collideWorldBounds = true;

    game.physics.enable(bullet3);
    bullet3.body.collideWorldBounds = true; 

    pauseLabel = game.add.text(canvasWidth - 100, 20, 'Pause', { font: '20px Arial', fill: '#fff' });
    pauseLabel.inputEnabled = true;
    pauseLabel.events.onInputUp.add(pause, this);
    game.input.onDown.add(pauseClick, this);

    jump = game.input.keyboard.addKey(Phaser.Keyboard.SPACEBAR);
    left = game.input.keyboard.addKey(Phaser.Keyboard.A);
    right = game.input.keyboard.addKey(Phaser.Keyboard.D);

    nnNetwork = new synaptic.Architect.Perceptron(2, 10, 10, 2);
    nnTraining = new synaptic.Trainer(nnNetwork);

    nnNetwork2 = new synaptic.Architect.Perceptron(2, 10, 10, 2);
    nnTraining2 = new synaptic.Trainer(nnNetwork2);

    nnNetwork3 = new synaptic.Architect.Perceptron(2, 10, 10, 2); 
    nnTraining3 = new synaptic.Trainer(nnNetwork3);
}

function neuralNetwork() {
    nnTraining.train(trainingData, { rate: 0.0003, iterations: 10000, shuffle: true });
}

function neuralNetworkAdvance() {
    nnTraining2.train(trainingData2, { rate: 0.0003, iterations: 10000, shuffle: true });
}
function neuralNetwork3() {
    nnTraining3.train(trainingData3, { rate: 0.0003, iterations: 10000, shuffle:true });
}
function trainingDataInput(param_input) {
    nnOutput = nnNetwork.activate(param_input);
    var air = Math.round(nnOutput[0] * 100);
    var ground = Math.round(nnOutput[1] * 100);
    return nnOutput[0] >= nnOutput[1];
}

function trainingDataInputBullet2(param_input) {
    nnOutput2 = nnNetwork2.activate(param_input);
    var progress = Math.round(nnOutput2[0] * 100);
    var still = Math.round(nnOutput2[1] * 100);
    return nnOutput2[0] >= nnOutput2[1];
}
function trainingDataInputBullet3(param_input) {
    nnOutput3 = nnNetwork3.activate(param_input);
    var horizontalMovement = Math.round(nnOutput3[0] * 100);
    var verticalMovement = Math.round(nnOutput3[1] * 100);

    return nnOutput3[0] >= nnOutput3[1];
}

function pause() {
    game.paused = true;
    menu = game.add.sprite(canvasWidth / 2, canvasHeight / 2, 'menu');
    menu.anchor.setTo(0.5, 0.5);
}

function pauseClick(event) {
    if (game.paused) {
        var menu_x1 = canvasWidth / 2 - 270 / 2,
            menu_x2 = canvasWidth / 2 + 270 / 2,
            menu_y1 = canvasHeight / 2 - 180 / 2,
            menu_y2 = canvasHeight / 2 + 180 / 2;

        var mouse_x = event.x,
            mouse_y = event.y;

        if (mouse_x > menu_x1 && mouse_x < menu_x2 && mouse_y > menu_y1 && mouse_y < menu_y2) {
            if (mouse_x >= menu_x1 && mouse_x <= menu_x2 && mouse_y >= menu_y1 && mouse_y <= menu_y1 + 90) {
                trainingComplete = false;
                trainingData = [];
                trainingData2 = [];
                autoMode = false;
                newGame = true;
            } else if (mouse_x >= menu_x1 && mouse_x <= menu_x2 && mouse_y >= menu_y1 + 90 && mouse_y <= menu_y2) {
                newGame = true;
                if (!trainingComplete) {
                    console.log('Entrenamiento: ', trainingData.length);
                    console.log('Entrenamiento2: ', trainingData2.length);
                    neuralNetwork();
                    neuralNetworkAdvance();
                    trainingComplete = true;
                }
                autoMode = true;
            }

            menu.destroy();
            resetVariables();
            game.paused = false;
        }
    }
}

function resetVariables() {
    player.body.velocity.x = 0;
    player.body.velocity.y = 0;
    player.body.position.x = 50;

    bullet.body.velocity.x = 0;
    bullet.position.x = canvasWidth - 100;

    bullet2.body.velocity.y = bulletSpeed2;
    bullet2.position.y = canvasHeight - 350;
    bulletD2 = false;
    bulletD = false;

    bullet3.body.velocity.x = -bulletSpeed3 * 5; 
    bullet3.body.velocity.y = bulletSpeed3;
    bullet3.position.x = canvasWidth - 100; 
    bullet3.position.y = 70; 
    bulletD3 = false; 
}

function jumpAction() {
    player.body.velocity.y = -270;
}

function moveRight() {
    if (player.body.position.x < 100)
        player.body.position.x += 10;
}

function moveRight2() {
    if (player.body.position.x < 100)
        player.body.position.x += 20;
}

function update() {
    if (newGame) {
        newGame = false;
        player.body.position.x = 50;
        ship2.position.x = 50;
        bullet2.position.x = 50;
        bullet2.position.y = canvasHeight - 350;
        bullet3.position.x = canvasWidth - 100; 
        bullet3.position.y = 70;
    }

    background.tilePosition.x -= 1;

    game.physics.arcade.collide(bullet, player, collisionHandler, null, this);
    game.physics.arcade.collide(bullet2, player, collisionHandler, null, this);
    game.physics.arcade.collide(bullet3, player, collisionHandler, null, this); 

    groundStatus = 1;
    airStatus = 0;
    progress = 0;
    still = 1;

    if (!player.body.onFloor()) {
        groundStatus = 0;
        airStatus = 1;
    }
    if (player.body.position.x > 50) {
        progress = 1;
        still = 0;
    }

    bullet2.body.velocity.y = bulletSpeed2;

    bulletDistance = Math.floor(player.position.x - bullet.position.x);
    bulletDistance2 = Math.floor(player.position.y - bullet2.position.y);
    bulletDistance2x = Math.floor(player.position.x - bullet2.position.x);
    bulletDistance3 = Math.floor(player.position.y - bullet3.position.y); 
    bulletDistance3x = Math.floor(player.position.x - bullet3.position.x);

    if (autoMode == false && right.isDown && player.body.onFloor()) {
        moveRight();
    }

    if (autoMode == false && jump.isDown && player.body.onFloor()) {
        jumpAction();
    }
    
    
    if (autoMode == true && bullet.position.x > 0 && player.body.onFloor()) {
        if (trainingDataInputBullet2([bulletDistance2, bulletSpeed2])) {
            if (bulletDistance2x === 0)
                moveRight2();
        }

        if (trainingDataInput([bulletDistance, bulletSpeed])) {
            if (bulletDistance2x === 0 && bulletDistance2 > 150)
                jumpAction();
            else if (bulletDistance2x > 0)
                jumpAction();
        }

        if (trainingDataInputBullet3([bulletDistance3, bulletSpeed3])) {
            if (bulletDistance3x === 0 )
                jumpAction();
            else if (bulletDistance3x > 0)
                jumpAction();
        }
    }
    
    if (bulletD == false) {
        fire();
    }

    if (bullet.position.x <= 0) {
        resetVariables();
    }

    if (bulletD2 == false) {
        fire2();
    }

    if (bullet3.position.y >= canvasHeight) {
        fire3();
    }

    if (autoMode == false && bullet.position.x > 0) {
        trainingData.push({
            'input': [bulletDistance, bulletSpeed],
            'output': [airStatus, groundStatus]
        });
    }

    if (autoMode == false && bullet2.position.y > 50) {
        trainingData2.push({
            'input': [bulletDistance2, bulletSpeed2],
            'output': [progress, still]
        });
    }
    if (autoMode == false && bullet3.position.y >= 0) {
        trainingData3.push({
            'input': [bulletDistance3, bulletSpeed3],
            'output': [progress, still] 
        });
    }
}

function fire() {
    bulletSpeed = -1 * randomSpeed(300, 400);
    bullet.body.velocity.y = 0;
    bullet.body.velocity.x = bulletSpeed;
    bulletD = true;
    bulletD2 = true;
}

function fire2() {
    bullet2.body.velocity.y = bulletSpeed2;
    bulletD2 = true;
}

function fire3() {
    var dx = player.position.x - bullet3.position.x;
    var dy = player.position.y - bullet3.position.y;
    var magnitude = Math.sqrt(dx * dx + dy * dy);
    
    bullet3.body.velocity.x = (dx / magnitude) * bulletSpeed3;
    bullet3.body.velocity.y = (dy / magnitude) * bulletSpeed3;
    
    bullet3.position.x = canvasWidth - 100;
    bullet3.position.y = canvasHeight - 50;
    bulletD3 = true;
}

function collisionHandler() {
    pause();
}

function randomSpeed(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

function render() {
    // Render code, if needed
}
