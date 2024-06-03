var w = 350;
var h = 350;
var jugador;
var fondo;
var bala;
var VOLVIENDOV = false;
var VOLVIENDOH = false;

var cursors;
var menu;

var estatusIzquierda;
var estatusDerecha;
var estatusArriba;
var estatusAbajo;
var estatusMovimiento;
var nnNetwork, nnEntrenamiento, nnSalida, datosEntrenamiento = [];
var modoAuto = false, eCompleto = false;

var JX = 200;
var JY = 200;

var autoMode = false;
var juego = new Phaser.Game(w, h, Phaser.CANVAS, '', { preload: preload, create: create, update: update, render: render });

function preload() {
    // Precargar imágenes y sprites
    juego.load.image('background', 'assets/game/background1.jpg');
    juego.load.spritesheet('mono', 'assets/sprites/altair.png', 32, 48);
    juego.load.image('menu', 'assets/game/menu.png');
    juego.load.image('bala', 'assets/sprites/purple_ball.png');
}

function create() {
    // Inicializar el sistema de físicas
    juego.physics.startSystem(Phaser.Physics.ARCADE);
    juego.physics.arcade.gravity.y = 0; // Sin gravedad para permitir movimiento libre
    juego.time.desiredFps = 30;

    // Crear el fondo y el jugador
    fondo = juego.add.tileSprite(0, 0, w, h, 'background');
    jugador = juego.add.sprite(w / 2, h / 2, 'mono');

    // Habilitar física para el jugador
    juego.physics.enable(jugador);
    jugador.body.collideWorldBounds = true; // Evitar que el jugador salga de los límites del mundo
    var corre = jugador.animations.add('corre', [8, 9, 10, 11]);
    jugador.animations.play('corre', 10, true);

    // Añadir la bala en la esquina superior derecha
    bala = juego.add.sprite(0, 0, 'bala');
    juego.physics.enable(bala);
    bala.body.collideWorldBounds = true; // Evitar que la bala salga de los límites del mundo
    bala.body.bounce.set(1); // Hacer que la bala rebote
    setRandomBalaVelocity(); // Establecer una velocidad inicial aleatoria para la bala

    // Añadir texto de pausa
    pausaL = juego.add.text(w - 100, 20, 'Pausa', { font: '20px Arial', fill: '#fff' });
    pausaL.inputEnabled = true;
    pausaL.events.onInputUp.add(pausa, self);
    juego.input.onDown.add(mPausa, self);

    // Creación de teclas de dirección
    cursors = juego.input.keyboard.createCursorKeys();

    // Crear red neuronal
    nnNetwork = new synaptic.Architect.Perceptron(3, 6, 6, 6, 5);
    nnEntrenamiento = new synaptic.Trainer(nnNetwork);
}

function enRedNeural() {
    // Entrenar la red neuronal con los datos de entrenamiento
    nnEntrenamiento.train(datosEntrenamiento, { rate: 0.0003, iterations: 10000, shuffle: true });
}

function datosVertical(param_entrada) {
    // Función para procesar datos verticales
    console.log("Entrada", param_entrada[0] + " " + param_entrada[1] + " " + param_entrada[2] + " " + param_entrada[3] + " " + param_entrada[4]);
    nnSalida = nnNetwork.activate(param_entrada);
    var Izq = Math.round(nnSalida[0] * 100);
    var Der = Math.round(nnSalida[1] * 100);
    var Arr = Math.round(nnSalida[2] * 100);
    var Aba = Math.round(nnSalida[3] * 100);
    var xde = Math.round(nnSalida[4] * 100);

    if (param_entrada[2] < 80) {
        if (Arr > 40 && Arr < 60) {
            return false;
        }
    }

    console.log("\n En la estatusArriba %: " + nnSalida[2] * 100 +
        "\n En la estatusAbajo %: " + nnSalida[3] * 100 +
        "\n En la estatusIzq %: " + nnSalida[0] * 100 +
        "\n En la estatusDer %: " + nnSalida[1] * 100 +
        "\n En movimiento %: " + nnSalida[4] * 100);
    console.log("OUTPUTS: " + nnSalida[2] >= nnSalida[3])
    return nnSalida[2] >= nnSalida[3];
}

function datosHorizontal(param_entrada) {
    // Función para procesar datos horizontales
    console.log("Entrada", param_entrada[0] + " " + param_entrada[1] + " " + param_entrada[2] + " " + param_entrada[3] + " " + param_entrada[4]);
    nnSalida = nnNetwork.activate(param_entrada);
    var Izq = Math.round(nnSalida[0] * 100);
    var Der = Math.round(nnSalida[1] * 100);
    var Arr = Math.round(nnSalida[2] * 100);
    var Aba = Math.round(nnSalida[3] * 100);
    var xde = Math.round(nnSalida[4] * 100);

    if (param_entrada[2] < 80) {
        if (Der > 40 && Der < 60) {
            return false;
        }
    }

    console.log("\n En la estatusArriba %: " + nnSalida[2] * 100 +
        "\n En la estatusAbajo %: " + nnSalida[3] * 100 +
        "\n En la estatusIzq %: " + nnSalida[0] * 100 +
        "\n En la estatusDer %: " + nnSalida[1] * 100 +
        "\n En movimiento %: " + nnSalida[4] * 100);
    console.log("OUTPUTS: " + nnSalida[2] >= nnSalida[3])
    return nnSalida[0] >= nnSalida[1];
}

function datosMovimiento(param_entrada) {
    // Función para procesar datos de movimiento
    console.log("Entrada", param_entrada[0] + " " + param_entrada[1] + " " + param_entrada[2] + " " + param_entrada[3] + " " + param_entrada[4]);
    nnSalida = nnNetwork.activate(param_entrada);
    var Izq = Math.round(nnSalida[0] * 100);
    var Der = Math.round(nnSalida[1] * 100);
    var Arr = Math.round(nnSalida[2] * 100);
    var Aba = Math.round(nnSalida[3] * 100);
    var xde = Math.round(nnSalida[4] * 100);

    if (param_entrada[2] < 80) {
        if (Der > 40 && Der < 60) {
            return false;
        }
    }

    console.log("\n En la estatusArriba %: " + nnSalida[2] * 100 +
        "\n En la estatusAbajo %: " + nnSalida[3] * 100 +
        "\n En la estatusIzq %: " + nnSalida[0] * 100 +
        "\n En la estatusDer %: " + nnSalida[1] * 100 +
        "\n La disque desta %: " + nnSalida[4] * 100);
    console.log("OUTPUTS: " + nnSalida[2] >= nnSalida[3])
    return nnSalida[4] * 100 >= 20;
}

function pausa() {
    // Pausar el juego
    juego.paused = true;
    menu = juego.add.sprite(w / 2, h / 2, 'menu'); // Añadir menú de pausa
    menu.anchor.setTo(0.5, 0.5);
}

function mPausa(event) {
    // Manejar la pausa del juego
    if (juego.paused) {
        var menu_x1 = w / 2 - 270 / 2, menu_x2 = w / 2 + 270 / 2,
            menu_y1 = h / 2 - 180 / 2, menu_y2 = h / 2 + 180 / 2;

        var mouse_x = event.x,
            mouse_y = event.y;

        if (mouse_x > menu_x1 && mouse_x < menu_x2 && mouse_y > menu_y1 && mouse_y < menu_y2) {
            if (mouse_x >= menu_x1 && mouse_x <= menu_x2 && mouse_y >= menu_y1 && mouse_y <= menu_y1 + 90) {
                eCompleto = false;
                datosEntrenamiento = [];
                modoAuto = false;
            } else if (mouse_x >= menu_x1 && mouse_x <= menu_x2 && mouse_y >= menu_y1 + 90 && mouse_y <= menu_y2) {
                if (!eCompleto) {
                    console.log("", "Entrenamiento " + datosEntrenamiento.length + " valores");
                    enRedNeural();
                    eCompleto = true;
                }
                modoAuto = true;
            }
            menu.destroy();
            reset();
            juego.paused = false;
        } else {
            menu.destroy();
            juego.paused = false;
        }
    }
}

function reset() {
    // Resetear el estado del jugador
    JX = 200;
    JY = 200;
    jugador.reset(JX, JY);
}

function update() {
    // Lógica de actualización del juego
    if (modoAuto) {
        var disx, disy, dx, dy, d, mi;
        disx = bala.body.x - jugador.body.x;
        disy = bala.body.y - jugador.body.y;
        dx = disx;
        dy = disy;

        var dis = Math.sqrt((dx * dx) + (dy * dy));

        estatusIzquierda = jugador.x > 32 ? 1 : 0;
        estatusDerecha = jugador.x < w - 32 ? 1 : 0;
        estatusArriba = jugador.y > 32 ? 1 : 0;
        estatusAbajo = jugador.y < h - 32 ? 1 : 0;
        estatusMovimiento = jugador.body.velocity.x != 0 || jugador.body.velocity.y != 0 ? 1 : 0;

        if (estatusIzquierda && estatusDerecha && estatusArriba && estatusAbajo) {
            if (Math.abs(dx) > Math.abs(dy)) {
                VOLVIENDOH = datosHorizontal([Math.abs(dx), Math.abs(dy), dis, estatusIzquierda, estatusDerecha, estatusArriba, estatusAbajo, estatusMovimiento]);
            } else {
                VOLVIENDOV = datosVertical([Math.abs(dx), Math.abs(dy), dis, estatusIzquierda, estatusDerecha, estatusArriba, estatusAbajo, estatusMovimiento]);
            }
        } else {
            if (estatusIzquierda && estatusDerecha) {
                VOLVIENDOH = datosHorizontal([Math.abs(dx), Math.abs(dy), dis, estatusIzquierda, estatusDerecha, estatusArriba, estatusAbajo, estatusMovimiento]);
            }
            if (estatusArriba && estatusAbajo) {
                VOLVIENDOV = datosVertical([Math.abs(dx), Math.abs(dy), dis, estatusIzquierda, estatusDerecha, estatusArriba, estatusAbajo, estatusMovimiento]);
            }
        }

        jugador.body.velocity.x = VOLVIENDOH ? 150 : -150;
        jugador.body.velocity.y = VOLVIENDOV ? 150 : -150;
    } else {
        if (cursors.left.isDown) {
            jugador.body.velocity.x = -150;
            datosEntrenamiento.push([Math.abs(jugador.body.x - bala.body.x), Math.abs(jugador.body.y - bala.body.y), Math.sqrt(Math.pow(Math.abs(jugador.body.x - bala.body.x), 2) + Math.pow(Math.abs(jugador.body.y - bala.body.y), 2)), estatusIzquierda, estatusDerecha, estatusArriba, estatusAbajo, estatusMovimiento]);
        }
        if (cursors.right.isDown) {
            jugador.body.velocity.x = 150;
            datosEntrenamiento.push([Math.abs(jugador.body.x - bala.body.x), Math.abs(jugador.body.y - bala.body.y), Math.sqrt(Math.pow(Math.abs(jugador.body.x - bala.body.x), 2) + Math.pow(Math.abs(jugador.body.y - bala.body.y), 2)), estatusIzquierda, estatusDerecha, estatusArriba, estatusAbajo, estatusMovimiento]);
        }
        if (cursors.up.isDown) {
            jugador.body.velocity.y = -150;
            datosEntrenamiento.push([Math.abs(jugador.body.x - bala.body.x), Math.abs(jugador.body.y - bala.body.y), Math.sqrt(Math.pow(Math.abs(jugador.body.x - bala.body.x), 2) + Math.pow(Math.abs(jugador.body.y - bala.body.y), 2)), estatusIzquierda, estatusDerecha, estatusArriba, estatusAbajo, estatusMovimiento]);
        }
        if (cursors.down.isDown) {
            jugador.body.velocity.y = 150;
            datosEntrenamiento.push([Math.abs(jugador.body.x - bala.body.x), Math.abs(jugador.body.y - bala.body.y), Math.sqrt(Math.pow(Math.abs(jugador.body.x - bala.body.x), 2) + Math.pow(Math.abs(jugador.body.y - bala.body.y), 2)), estatusIzquierda, estatusDerecha, estatusArriba, estatusAbajo, estatusMovimiento]);
        }
    }
    if (!modoAuto && !cursors.left.isDown && !cursors.right.isDown && !cursors.up.isDown && !cursors.down.isDown) {
        jugador.body.velocity.x = 0;
        jugador.body.velocity.y = 0;
    }
}

function render() {
    // Renderizar elementos del juego
    juego.debug.text('Modo Auto: ' + (modoAuto ? 'Activado' : 'Desactivado'), 20, 30);
    juego.debug.text('Entrenamiento: ' + datosEntrenamiento.length + ' valores', 20, 50);
}

function setRandomBalaVelocity() {
    // Establecer una velocidad aleatoria para la bala
    var min = -300;
    var max = 300;
    bala.body.velocity.x = Math.random() * (max - min) + min;
    bala.body.velocity.y = Math.random() * (max - min) + min;
}


function setRandomBalaVelocity() {
    var speed = 300;
    var angle = juego.rnd.angle(); // Obtener un ángulo aleatorio
    bala.body.velocity.set(Math.cos(angle) * speed, Math.sin(angle) * speed);
}

function update() {
    fondo.tilePosition.x -= 1; // Mover el fondo para crear efecto de desplazamiento


    if (!autoMode) {
        // Resetear velocidad del jugador
        jugador.body.velocity.x = 0;
        jugador.body.velocity.y = 0;

        // Movimiento del jugador con teclas de dirección
        if (cursors.left.isDown) {
            jugador.body.velocity.x = -300; // Mover a la izquierda
        } else if (cursors.right.isDown) {
            jugador.body.velocity.x = 300; // Mover a la derecha
        }

        if (cursors.up.isDown) {
            jugador.body.velocity.y = -300; // Mover hacia arriba
        } else if (cursors.down.isDown) {
            jugador.body.velocity.y = 300; // Mover hacia abajo
        }
    } 

   

    // Colisionar la bala con el jugador
    juego.physics.arcade.collide(bala, jugador, colisionH, null, this);

    // Calcular la distancia entre la bala y el jugador
    var dx = bala.x - jugador.x;
    var dy = bala.y - jugador.y;
    var distancia = Math.sqrt(dx * dx + dy * dy); // Fórmula de distancia euclidiana, verifica las coordenadas x,y

    // Determinar el cuadrante en el que se encuentra la bala respecto al jugador


    
    estatusIzquierda = 0;
    estatusDerecha = 0;
    estatusArriba = 0;
    estatusAbajo = 0;
    estatusMovimiento = 0;


    if(!autoMode) {
        // Si la bala está a la derecha, moverse a la izquierda, y viceversa
        if (dx > 0) {
            estatusIzquierda = 1;
            estatusMovimiento = 1;
        } else {
            estatusDerecha = 1 // Mover a la derecha
        }
    
        // Si la bala está abajo, moverse hacia arriba, y viceversa
        if (dy > 0) {
            estatusArriba = 1; // Mover hacia arriba
        } else {
            estatusAbajo = 1; // Mover hacia abajo
        }


        if(jugador.body.velocity.x != 0 || jugador.body.velocity.y != 0) {
            estatusMovimiento = 1;
        }else{
            estatusMovimiento = 0;
        }
    }

    
    console.log("DATOS MOVIMIENTO: "+ datosMovimiento([dx, dy, distancia, JX, JY]));
    if(autoMode && datosMovimiento([dx, dy, distancia, JX, JY])) {


        if(distancia <=150){
        // Si la bala está abajo, moverse hacia arriba, y viceversa
        console.log("RETURN DEL METODO VERTICAL: " + datosVertical([dx, dy, distancia, JX, JY])
                 +"\nRETURN DEL METODO HORIZONTAL: " + datosHorizontal([dx, dy, distancia, JX, JY]));

        if (datosVertical([dx, dy, distancia, JX, JY]) && !VOLVIENDOV) {
            // Mover hacia arriba si datosVertical es true
            jugador.body.velocity.y -= 35;
            console.log("ARRIBA");
        } else if (!datosVertical([dx, dy, distancia, JX, JY]) && !VOLVIENDOV && distancia <= 95){
            // Mover hacia abajo si datosVertical es false
            jugador.body.velocity.y += 35;
            console.log("ABAJO");
        } 

        if (datosHorizontal([dx, dy, distancia, JX, JY]) && !VOLVIENDOH) {
            // Mover hacia arriba si datosHorizontal es true
            jugador.body.velocity.x -= 35;
            console.log("IZQUIERDA");
        } else if (!datosHorizontal([dx, dy, distancia, JX, JY]) && !VOLVIENDOH && distancia <= 95){
            // Mover hacia abajo si datosHorizontal es false
            jugador.body.velocity.x += 35;
            console.log("DERECHA");
        }


    
            // Ajustar la velocidad para que vuelva lentamente al centro si no está en movimiento
            if (jugador.x > 300) {
                jugador.body.velocity.x = -350; // Mover lentamente hacia arriba
                console.log("VOLVIENDOH AL CENTRO HACIA LA IZQUIERDA");
                VOLVIENDOH = true;
            } else if (jugador.x < 100) {
                jugador.body.velocity.x = 350; // Mover lentamente hacia abajo
                console.log("VOLVIENDOH AL CENTRO HACIA LA DERECHA");
                VOLVIENDOH = true;
            } else if(VOLVIENDOH && jugador.x > 150 && jugador.x < 250) {
                jugador.body.velocity.x = 0;
                VOLVIENDOH = false;
            } else if(datosHorizontal([dx, dy, distancia, JX, JY]) && jugador.body.velocity.x != 0){
                VOLVIENDOH = false;
                VOLVIENDOV = false;
            }


            // Ajustar la velocidad para que vuelva lentamente al centro si no está en movimiento
            if (jugador.y > 300) {
                jugador.body.velocity.y = -350; // Mover lentamente hacia arriba
                console.log("VOLVIENDOV AL CENTRO HACIA ARRIBA");
                VOLVIENDOV = true;
            } else if (jugador.y < 100) {
                jugador.body.velocity.y = 350; // Mover lentamente hacia abajo
                console.log("VOLVIENDOV AL CENTRO HACIA ABAJO");
                VOLVIENDOV = true;
            } else if(VOLVIENDOV && jugador.y > 150 && jugador.y < 250) {
                jugador.body.velocity.y = 0;
                VOLVIENDOV = false;
            } else if(datosVertical([dx, dy, distancia, JX, JY]) && jugador.body.velocity.y != 0){
                VOLVIENDOH = false;
                VOLVIENDOV = false;
                VOLVIENDOH = false;
                VOLVIENDOV = false;
            }
            
        }else if (distancia >= 200){
            jugador.body.velocity.y = 0;
            jugador.body.velocity.x = 0;

        }

    }

   

    if( modoAuto ==false  && bala.position.x > 0 ){
        JX = jugador.x;
        JY = jugador.y;

        datosEntrenamiento.push({
                'input' :  [dx , dy, distancia, JX, JY ],
                'output':  [estatusIzquierda , estatusDerecha, estatusArriba, estatusAbajo, estatusMovimiento]  
        });

        console.log(
                    "DX: ", dx + "\n"+
                    "DY: ", dy + "\n"+
                    "D: ", distancia + "\n"+
                    "JX: ", jugador.x + "\n"+
                    "JY: ", jugador.y + "\n"+
                    "BX: ", bala.x + "\n"+
                    "BY: ", bala.y + "\n"
                );
        console.log(
                    "estatusIzquierda: ", estatusIzquierda + "\n"+
                    "estatusDerecha: ", estatusDerecha + "\n"+
                    "estatusArriba: ", estatusArriba + "\n"+
                    "estatusAbajo: ", estatusAbajo + "\n"+
                    "estatusMovimiento: ", estatusMovimiento + "\n"
                );
   }
}

function colisionH() {
    autoMode = true; 
    pausa(); 
}
function render() {
}