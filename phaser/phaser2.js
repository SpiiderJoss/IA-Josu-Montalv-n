var w=500;
var h=400;

var jugador;
var fondo;
var bala;
var menu;
var cursors;

var key_der;
var key_izq;
var key_up;
var key_down;

var reset;
var velocidadBala;
var coordenada_X, coordenada_Y, dis_bala_jugador;

var estado_der;
var estado_izq;
var estado_arriba;
var estado_abajo;

var jugadorPosInicial = { x: 230, y: 200 };
var balaPosInicial = { x: w-10, y: h-10 };

var timer = 0;
var regresar = true;

var nnNetwork , nnEntrenamiento, nnSalida, datosEntrenamiento=[];
var modoAuto = false, eCompleto=false;

var juego = new Phaser.Game(w, h, Phaser.CANVAS, '', { preload: preload, create: create, update: update, render:render});

function preload() { 
    juego.load.image('fondo', 'assets/game/background1.jpg');
    juego.load.spritesheet('mono', 'assets/sprites/altair.png', 32, 48);
    juego.load.image('menu', 'assets/game/menu.png');
    juego.load.image('bala', 'assets/sprites/purple_ball.png');
}


function create() { 
    cursors = this.input.keyboard.createCursorKeys(); 
    juego.physics.startSystem(Phaser.Physics.ARCADE); 
    juego.time.desiredFps = 30; 
    fondo = juego.add.tileSprite(0, 0, w, h, 'fondo');
    bala = juego.add.sprite(balaPosInicial.x, balaPosInicial.y, 'bala');
    jugador = juego.add.sprite(jugadorPosInicial.x, jugadorPosInicial.y, 'mono');
    juego.physics.enable(jugador); 
    jugador.body.collideWorldBounds = true; 
    jugador.body.gravity.y = 0; 
    juego.physics.enable(bala); 
    bala.body.collideWorldBounds = true; 
    bala.body.bounce.set(1); 
    bala.body.velocity.set(600); 

    pausaL = juego.add.text(w - 100, 20, 'Pausa', { font: '20px Arial', fill: '#fff' }); 
    pausaL.inputEnabled = true; 
    pausaL.events.onInputUp.add(pausa, self); 
    juego.input.onDown.add(mPausa, self); 

    key_der = juego.input.keyboard.addKey(Phaser.Keyboard.RIGHT);
    key_izq = juego.input.keyboard.addKey(Phaser.Keyboard.LEFT);
    key_up = juego.input.keyboard.addKey(Phaser.Keyboard.UP);
    key_down = juego.input.keyboard.addKey(Phaser.Keyboard.DOWN);

    nnNetwork =  new synaptic.Architect.Perceptron(3,6,6,4);   // (3,3,6,4) (3,6,6,4) (3,3,6,3,4) (3,6,6,6,4)
    nnEntrenamiento = new synaptic.Trainer(nnNetwork); 
}

function enRedNeural(){ 
    nnEntrenamiento.train(datosEntrenamiento, {rate: 0.0003, iterations: 15000, shuffle: true});
    console.log("Entrenamiento Terminado")
}

function datosDeEntrenamiento(param_entrada){ 
    console.log("Input: dis_bala_jugador: "+param_entrada[0]+" coordenada_X: "+param_entrada[1]+" coordenada_Y: "+param_entrada[2]); 
    nnSalida = nnNetwork.activate(param_entrada);
   
    var porcentajeArriba = Math.round(nnSalida[0] * 100);
    var porcentajeAbajo = Math.round(nnSalida[1] * 100);
    var porcentajeDerecha = Math.round(nnSalida[2] * 100);
    var porcentajeIzquierda = Math.round(nnSalida[3] * 100);

    console.log("Arriba: "+porcentajeArriba + " Abajo: "+porcentajeAbajo +" Der: "+porcentajeDerecha + " Izq: "+porcentajeIzquierda);
    
    var maxValue = Math.max(nnSalida[0], nnSalida[1], nnSalida[2], nnSalida[3]);
    var minValue = Math.min(nnSalida[0], nnSalida[1], nnSalida[2], nnSalida[3]);
    var umbral = maxValue - minValue;           
    var accion = nnSalida.indexOf(maxValue); 

    if (umbral < 0.01) {  
        return 4; 
    }
    return accion;   
}


function pausa() {
    juego.paused = true;
    menu = juego.add.sprite(w/2, h/2, 'menu');
    menu.anchor.setTo(0.5, 0.5);
}


function mPausa(event){     
    if(juego.paused){ 
            var menu_x1 = w/2 - 270/2, menu_x2 = w/2 + 270/2,
            menu_y1 = h/2 - 180/2, menu_y2 = h/2 + 180/2;
            var mouse_x = event.x , mouse_y = event.y;
     if(mouse_x > menu_x1 && mouse_x < menu_x2 && mouse_y > menu_y1 && mouse_y < menu_y2 ){
            if(mouse_x >=menu_x1 && mouse_x <=menu_x2 && mouse_y >=menu_y1 && mouse_y <=menu_y1+90){  
                eCompleto=false; 
                datosEntrenamiento = []; 
                modoAuto = false; 
            }else if (mouse_x >=menu_x1 && mouse_x <=menu_x2 && mouse_y >=menu_y1+90 && mouse_y <=menu_y2) { 
                if(!eCompleto) { 
                    console.log("Datos de Entrenamiento: "+ datosEntrenamiento.length +" valores" );
                    enRedNeural(); 
                    eCompleto=true; 
                }
                modoAuto = true; 
            }
            resetVariables();  
            menu.destroy(); 
            juego.paused = false; 

        }
    }
}

function resetVariables(){
    velocidadBala = 450;
    bala.body.velocity.x = velocidadBala;
    bala.body.velocity.y = -velocidadBala;
    bala.position.x = balaPosInicial.x; 
    bala.position.y = balaPosInicial.y; 
    key_der.isDown = true;
    key_down.isDown=false;
    key_izq.isDown=false;
    key_up.isDown=false;
    jugador.position.x=230;
    jugador.position.y=200;
}


function derecha(){
    jugador.position.x = 380;
    jugador.position.y=200;
    estado_der=1;
    estado_izq=0;
    regresar=false;
}

function izquierda(){
    jugador.position.x = 80;
    jugador.position.y=200;
    estado_der=0;
    estado_izq=1;
    regresar=false;
}

function arriba(){
    jugador.position.x =230;
    jugador.position.y=80;
    estado_arriba=1;
    estado_abajo=0;
    regresar=false;
}
function abajo(){
    jugador.position.x =230;
    jugador.position.y=320;
    estado_arriba=0;
    estado_abajo=1;
    regresar=false;
}


function update() { 

    if(key_der && key_izq && key_down && key_up) {
        jugador.position.x=230;
        jugador.position.y=200;
    }
    
    if (!regresar) {
         timer += 1; 
    }
    posicion_inical();

    estado_der = 0;
    estado_izq = 0;
    estado_arriba = 0;
    estado_abajo = 0;
    
    coordenada_X = bala.position.x - jugador.position.x;  // Diferencia de las coordenadas para detectar el cuadrante
    coordenada_Y = bala.position.y - jugador.position.y;  // Diferencia de las coordenadas para detectar el cuadrante 
    dis_bala_jugador = Math.sqrt(coordenada_X * coordenada_X + coordenada_Y * coordenada_Y); // Fórmula de distancia bala-jugador

    if (key_der.isDown) { 
        jugador.position.x = 380;
        jugador.position.y = 200;
        estado_der = 1;
        estado_izq = 0;
        regresar = false;
    } else if (key_izq.isDown) {
        jugador.position.x = 80;
        jugador.position.y = 200;
        estado_der = 0;
        estado_izq = 1;
        regresar = false;
    }
    if (key_up.isDown) { 
        jugador.position.x = 230;
        jugador.position.y = 80;
        estado_arriba = 1;
        estado_abajo = 0;
        regresar = false;
    } else if (key_down.isDown) {
        jugador.position.x = 230;
        jugador.position.y = 320;
        estado_arriba = 0;
        estado_abajo = 1;
        regresar = false;
    }
        
    if (modoAuto == true && dis_bala_jugador <= 200) {
        var decision = datosDeEntrenamiento([dis_bala_jugador, coordenada_X, coordenada_Y]);
        if (decision === 0) arriba();
        else if (decision === 1) abajo();
        else if (decision === 2) derecha();
        else if (decision === 3) izquierda();    
    }

    juego.physics.arcade.collide(bala, jugador, colisionH, null, this);
    

    if (modoAuto == false) {
        var nuevoDatoEntrenamiento = { 
            'input': [dis_bala_jugador, coordenada_X, coordenada_Y], 
            'output': [estado_arriba, estado_abajo, estado_der, estado_izq]
        };
        datosEntrenamiento.push(nuevoDatoEntrenamiento);
        console.log({
            dis_bala_jugador: dis_bala_jugador,
            coordenada_X: coordenada_X,
            coordenada_Y: coordenada_Y,
            estado_arriba: estado_arriba,
            estado_abajo: estado_abajo,
            estado_der: estado_der,
            estado_izq: estado_izq
        });
    }
}

function posicion_inical(){
    if (timer > 25) {
        jugador.position.x = Phaser.Math.linear(jugador.position.x, 230, 0.5); 
        jugador.position.y = Phaser.Math.linear(jugador.position.y, 200, 0.5);
        if (Math.abs(jugador.position.x - 230) < 1 && Math.abs(jugador.position.y - 200) < 1) {
            regresar = true;
            timer = 0;
            jugador.position.x = 230;
            jugador.position.y = 200;
        }
    }
}

function colisionH(){  
    pausa();     
}

function render(){
  juego.debug.text(
    "Modo Auto: " + (modoAuto ? "Activado" : "Desactivado"),
    20,
    30
  );
  juego.debug.text(
    "Entrenamiento: " + datosEntrenamiento.length + " valores",
    20,
    50
  );
}