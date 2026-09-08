## RL Example


* https://rcalix1.github.io/artAI/ASCII3d/RL/ascii_city_rl_clear.html


```


import random

# ============================================================
# PEDESTRIAN Q-LEARNING
#
# Agent:
#       WAIT = 0
#       WALK = 1
#
# Output:
#       pedestrian_policy.js
#
# Run:
#       python train_pedestrian_rl.py
# ============================================================

EPISODES = 50000

ALPHA   = 0.15
GAMMA   = 0.95
EPSILON = 0.20

WAIT = 0
WALK = 1

N_CAR_STATES = 5
N_PED_STATES = 11


# ============================================================
# Q TABLE
# ============================================================

Q = {}

for car_state in range(N_CAR_STATES):

    for ped_state in range(N_PED_STATES):

        Q[(car_state, ped_state)] = [0.0, 0.0]


# ============================================================
# CAR STATE
#
# Car moves from negative X toward positive X.
# Crossing is at X = 0.
#
# 0 = very close
# 1 = close
# 2 = medium
# 3 = far
# 4 = passed
# ============================================================

def get_car_state(car_x):

    if car_x > 5:
        return 4

    distance = abs(car_x)

    if distance < 4:
        return 0

    if distance < 10:
        return 1

    if distance < 20:
        return 2

    return 3


# ============================================================
# TRAIN
# ============================================================

successes  = 0
collisions = 0


for episode in range(EPISODES):

    car_x = random.uniform(-35, -10)

    # pedestrian progress
    #
    # 0  = starting sidewalk
    # 1
    # 2
    # 3-7 = road
    # 8
    # 9
    # 10 = opposite sidewalk

    ped = 0

    done = False


    for step in range(100):

        car_state = get_car_state(car_x)

        state = (
            car_state,
            ped
        )


        # ====================================================
        # CHOOSE ACTION
        # ====================================================

        if random.random() < EPSILON:

            action = random.choice(
                [WAIT, WALK]
            )

        else:

            if (
                Q[state][WALK] >
                Q[state][WAIT]
            ):

                action = WALK

            else:

                action = WAIT


        # ====================================================
        # UPDATE WORLD
        # ====================================================

        old_ped = ped


        if action == WALK:

            ped += 1

            if ped > 10:
                ped = 10


        # car moves toward crossing

        car_x += 1.5


        # ====================================================
        # COLLISION
        # ====================================================

        pedestrian_in_road = (
            3 <= ped <= 7
        )

        car_at_crossing = (
            abs(car_x) < 3
        )


        collision = (
            pedestrian_in_road
            and
            car_at_crossing
        )


        success = (
            ped >= 10
        )


        # ====================================================
        # REWARD
        # ====================================================

        reward = -0.1


        if collision:

            reward = -100.0

            collisions += 1

            done = True


        elif success:

            reward = 100.0

            successes += 1

            done = True


        elif ped > old_ped:

            reward += 0.5


        # ====================================================
        # Q LEARNING
        #
        # Q(s,a) =
        #
        # Q(s,a) +
        # alpha [
        #
        # reward +
        # gamma max Q(s',a')
        # -
        # Q(s,a)
        #
        # ]
        # ====================================================

        next_car_state = get_car_state(
            car_x
        )

        next_state = (
            next_car_state,
            ped
        )


        old_q = Q[state][action]


        if done:

            target = reward

        else:

            target = (
                reward
                +
                GAMMA *
                max(Q[next_state])
            )


        Q[state][action] = (
            old_q
            +
            ALPHA *
            (
                target
                -
                old_q
            )
        )


        if done:
            break


    # reduce exploration during training

    if (
        episode > 0
        and
        episode % 5000 == 0
    ):

        EPSILON *= 0.85


# ============================================================
# CREATE FINAL POLICY
#
# policy[car_state][ped_state]
#
# 0 = WAIT
# 1 = WALK
# ============================================================

policy = []


for car_state in range(
    N_CAR_STATES
):

    row = []


    for ped_state in range(
        N_PED_STATES
    ):

        q_wait = Q[
            (car_state, ped_state)
        ][WAIT]

        q_walk = Q[
            (car_state, ped_state)
        ][WALK]


        if q_walk > q_wait:

            action = WALK

        else:

            action = WAIT


        row.append(action)


    policy.append(row)


# ============================================================
# WRITE JAVASCRIPT POLICY
# ============================================================

with open(
    "pedestrian_policy.js",
    "w"
) as f:

    f.write(
        "// GENERATED BY PYTHON RL TRAINING\n"
    )

    f.write(
        "// 0 = WAIT, 1 = WALK\n\n"
    )

    f.write(
        "const PEDESTRIAN_POLICY = [\n"
    )


    for row in policy:

        f.write(
            "    "
            +
            str(row)
            +
            ",\n"
        )


    f.write(
        "];\n"
    )


# ============================================================
# RESULTS
# ============================================================

print()
print("====================================")
print("RL TRAINING COMPLETE")
print("====================================")

print(
    "Episodes:",
    EPISODES
)

print(
    "Successful crossings:",
    successes
)

print(
    "Collisions during training:",
    collisions
)

print()

print(
    "Created pedestrian_policy.js"
)

print()

print(
    "0 = WAIT"
)

print(
    "1 = WALK"
)

print()


for car_state, row in enumerate(
    policy
):

    print(
        "Car state",
        car_state,
        ":",
        row
    )


print()
print("====================================")
print("POLICY READY FOR ASCII CITY")
print("====================================")




```


and the html 


```


<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>ASCII City - RL Pedestrian</title>

<style>
html, body {
    margin: 0;
    background: #000;
    overflow: hidden;
}

canvas {
    display: block;
    width: 100vw;
    height: 100vh;
    background: #000;
}
</style>
</head>

<body>

<canvas id="canvas"></canvas>

<!-- GENERATED BY train_pedestrian_rl.py -->
<script src="pedestrian_policy.js"></script>

<script>

// ============================================================
// ASCII CITY - RL PEDESTRIAN EXPERIMENT
//
// 1 = RANDOM / UNTRAINED AGENT
// 2 = TRAINED RL AGENT
//
// Python trains the policy.
// JavaScript executes the learned policy.
//
// NO LIBRARIES
// NO WEBGL
// NO TEXTURES
// ============================================================


const canvas =
    document.getElementById("canvas");

const ctx =
    canvas.getContext("2d");


const CELL_W = 8;
const CELL_H = 13;

let W,H,COLS,ROWS;

let buffer;
let depthBuffer;

const SHADES = " .,:;-+=*#%@";


// ============================================================
// RESIZE
// ============================================================

function resize() {

    W = canvas.width =
        window.innerWidth;

    H = canvas.height =
        window.innerHeight;

    COLS =
        Math.floor(W/CELL_W);

    ROWS =
        Math.floor(H/CELL_H);

    ctx.font =
        CELL_H+"px monospace";

    ctx.textBaseline="top";
}

window.addEventListener(
    "resize",
    resize
);

resize();


// ============================================================
// MATH
// ============================================================

function clamp(v,a,b) {

    return Math.max(
        a,
        Math.min(b,v)
    );
}


function hash(n) {

    let x =
        Math.sin(
            n*12.9898+78.233
        )*43758.5453;

    return x-Math.floor(x);
}


// ============================================================
// WORLD
// ============================================================

const ROAD_HALF = 6;

const INTERSECTION_Z = 55;

const VIEW_DISTANCE = 120;


// ============================================================
// CAMERA
//
// Fixed camera.
// We deliberately keep the RL experiment visible.
// ============================================================

const camera = {

    x: -7.5,
    y: 2.0,
    z: 31,

    yaw: 0
};


// ============================================================
// BUILDINGS
// ============================================================

let buildings=[];


function addBuilding(
    x1,x2,
    z1,z2,
    height,
    seed
) {

    buildings.push({

        x1,x2,

        y1:0,
        y2:height,

        z1,z2,

        seed
    });
}


function generateCity() {

    buildings=[];

    let seed=1;


    // buildings along cross street

    for(
        let x=-100;
        x<100;
        x+=18
    ) {

        if(
            x>-18 &&
            x<18
        )
            continue;


        addBuilding(

            x,
            x+14,

            INTERSECTION_Z-24,
            INTERSECTION_Z-11,

            9+hash(seed)*18,

            seed++
        );


        addBuilding(

            x,
            x+14,

            INTERSECTION_Z+11,
            INTERSECTION_Z+24,

            9+hash(seed)*18,

            seed++
        );
    }
}

generateCity();


// ============================================================
// RAY BOX
// ============================================================

function rayBox(
    ox,oy,oz,
    dx,dy,dz,
    box
) {

    let tmin=-Infinity;
    let tmax=Infinity;

    let nx=0;
    let ny=0;
    let nz=0;


    // X

    if(Math.abs(dx)<0.000001) {

        if(
            ox<box.x1 ||
            ox>box.x2
        )
            return null;

    } else {

        let t1=(box.x1-ox)/dx;
        let t2=(box.x2-ox)/dx;

        let n=-1;

        if(t1>t2) {

            [t1,t2]=[t2,t1];

            n=1;
        }

        if(t1>tmin) {

            tmin=t1;

            nx=n;
            ny=0;
            nz=0;
        }

        tmax=Math.min(tmax,t2);

        if(tmin>tmax)
            return null;
    }


    // Y

    if(Math.abs(dy)<0.000001) {

        if(
            oy<box.y1 ||
            oy>box.y2
        )
            return null;

    } else {

        let t1=(box.y1-oy)/dy;
        let t2=(box.y2-oy)/dy;

        let n=-1;

        if(t1>t2) {

            [t1,t2]=[t2,t1];

            n=1;
        }

        if(t1>tmin) {

            tmin=t1;

            nx=0;
            ny=n;
            nz=0;
        }

        tmax=Math.min(tmax,t2);

        if(tmin>tmax)
            return null;
    }


    // Z

    if(Math.abs(dz)<0.000001) {

        if(
            oz<box.z1 ||
            oz>box.z2
        )
            return null;

    } else {

        let t1=(box.z1-oz)/dz;
        let t2=(box.z2-oz)/dz;

        let n=-1;

        if(t1>t2) {

            [t1,t2]=[t2,t1];

            n=1;
        }

        if(t1>tmin) {

            tmin=t1;

            nx=0;
            ny=0;
            nz=n;
        }

        tmax=Math.min(tmax,t2);

        if(tmin>tmax)
            return null;
    }


    if(tmin<0)
        return null;


    return {

        t:tmin,

        nx,ny,nz
    };
}


// ============================================================
// GROUND
// ============================================================

function rayGround(
    ox,oy,oz,
    dx,dy,dz
) {

    if(dy>=0)
        return null;

    let t=-oy/dy;

    if(t<=0)
        return null;


    return {

        t,

        x:ox+dx*t,

        z:oz+dz*t
    };
}


// ============================================================
// SHADE
// ============================================================

function shade(b) {

    b=clamp(b,0,1);

    return SHADES[
        Math.floor(
            b*(SHADES.length-1)
        )
    ];
}


// ============================================================
// TRACE
// ============================================================

function traceRay(dx,dy,dz) {

    let closest=Infinity;

    let hit=null;


    for(let box of buildings) {

        let r=rayBox(

            camera.x,
            camera.y,
            camera.z,

            dx,dy,dz,

            box
        );


        if(
            r &&
            r.t<closest &&
            r.t<VIEW_DISTANCE
        ) {

            closest=r.t;

            hit={

                type:"building",

                box,

                ...r
            };
        }
    }


    let g=rayGround(

        camera.x,
        camera.y,
        camera.z,

        dx,dy,dz
    );


    if(
        g &&
        g.t<closest &&
        g.t<VIEW_DISTANCE
    ) {

        hit={

            type:"ground",

            ...g
        };

        closest=g.t;
    }


    return hit;
}


// ============================================================
// GROUND TYPE
// ============================================================

function groundType(x,z) {

    const crossRoad =

        Math.abs(
            z-INTERSECTION_Z
        ) < ROAD_HALF;


    if(crossRoad)
        return "road";


    if(
        Math.abs(
            z-INTERSECTION_Z
        ) < ROAD_HALF+3
    )
        return "sidewalk";


    return "ground";
}


// ============================================================
// GROUND CHARACTER
// ============================================================

function groundCharacter(
    x,z,distance
) {

    let type=
        groundType(x,z);


    if(type==="road") {

        // center line

        if(
            Math.abs(
                z-INTERSECTION_Z
            ) < .12
        ) {

            if(
                Math.floor(x/3)%2===0
            )
                return "-";
        }

        return shade(.16);
    }


    if(type==="sidewalk") {

        let lineX =
            Math.abs((x*1.2)%1)<.06;

        let lineZ =
            Math.abs((z*1.2)%1)<.06;

        return shade(
            lineX||lineZ
            ? .32
            : .24
        );
    }


    return shade(.10);
}


// ============================================================
// PROJECT 3D POINT
// ============================================================

function projectPoint(x,y,z) {

    let px=x-camera.x;
    let py=y-camera.y;
    let pz=z-camera.z;


    let cos=
        Math.cos(camera.yaw);

    let sin=
        Math.sin(camera.yaw);


    let rx=
        px*cos-pz*sin;

    let rz=
        px*sin+pz*cos;


    if(rz<=.2)
        return null;


    let focal=
        COLS*.72;


    return {

        col:
            COLS/2+
            rx/rz*focal,

        row:
            ROWS/2-
            py/rz*focal*.52,

        depth:rz
    };
}


// ============================================================
// BUFFER WRITE
// ============================================================

function setChar(
    col,row,
    character,
    depth
) {

    col=Math.floor(col);
    row=Math.floor(row);


    if(
        col<0 ||
        col>=COLS ||
        row<0 ||
        row>=ROWS
    )
        return;


    let index=
        row*COLS+col;


    if(
        depth <
        depthBuffer[index]
    ) {

        buffer[index]=character;

        depthBuffer[index]=depth;
    }
}


// ============================================================
// CAR
//
// Moves left -> right through crossing.
// ============================================================

let car={

    x:-40,

    z:INTERSECTION_Z,

    speed:7
};


// ============================================================
// RL PEDESTRIAN
//
// Starts on near sidewalk.
//
// progress:
//      0  = start
//      10 = crossed
// ============================================================

let agent={

    x:0,

    progress:0,

    z:INTERSECTION_Z-9,

    action:0,

    decisionTimer:0
};


let mode="RANDOM";

let safeCrossings=0;

let collisions=0;

let message="";

let messageTimer=0;


// ============================================================
// KEYBOARD
// ============================================================

window.addEventListener(
    "keydown",
    function(e) {

        if(e.key==="1") {

            mode="RANDOM";

            resetEpisode();
        }


        if(e.key==="2") {

            mode="TRAINED";

            resetEpisode();
        }
    }
);


// ============================================================
// CAR STATE
//
// MUST MATCH PYTHON
// ============================================================

function getCarState() {

    if(car.x>5)
        return 4;


    let distance=
        Math.abs(car.x);


    if(distance<4)
        return 0;

    if(distance<10)
        return 1;

    if(distance<20)
        return 2;

    return 3;
}


// ============================================================
// AGENT DECISION
// ============================================================

function chooseAction() {

    if(mode==="RANDOM") {

        return (
            Math.random()<.5
            ? 0
            : 1
        );
    }


    let cs=
        getCarState();


    let ps=
        Math.round(
            agent.progress
        );


    ps=
        clamp(
            ps,
            0,
            10
        );


    return (
        PEDESTRIAN_POLICY[
            cs
        ][ps]
    );
}


// ============================================================
// RESET
// ============================================================

function resetEpisode() {

    car.x=
        -35-Math.random()*15;


    agent.progress=0;

    agent.z=
        INTERSECTION_Z-9;

    agent.action=0;

    agent.decisionTimer=0;
}


// ============================================================
// WORLD UPDATE
// ============================================================

function updateWorld(dt) {

    if(messageTimer>0) {

        messageTimer-=dt;

        return;
    }


    // --------------------------------------------------------
    // CAR
    // --------------------------------------------------------

    car.x +=
        car.speed*dt;


    // --------------------------------------------------------
    // AGENT DECISION
    //
    // decision roughly twice per second
    // --------------------------------------------------------

    agent.decisionTimer -= dt;


    if(agent.decisionTimer<=0) {

        agent.action=
            chooseAction();

        agent.decisionTimer=.45;
    }


    // --------------------------------------------------------
    // WALK
    // --------------------------------------------------------

    if(agent.action===1) {

        agent.progress +=
            1.45*dt;


        agent.progress=
            clamp(
                agent.progress,
                0,
                10
            );


        // map progress across road

        agent.z =
            INTERSECTION_Z
            -9
            +
            agent.progress*1.8;
    }


    // --------------------------------------------------------
    // COLLISION
    // --------------------------------------------------------

    let inRoad=

        agent.z >
        INTERSECTION_Z-ROAD_HALF

        &&

        agent.z <
        INTERSECTION_Z+ROAD_HALF;


    let carNear=

        Math.abs(
            car.x-agent.x
        ) < 2.5;


    if(
        inRoad &&
        carNear
    ) {

        collisions++;

        message=
            "!!! COLLISION !!!";

        messageTimer=2.0;

        setTimeout(
            resetEpisode,
            2000
        );

        return;
    }


    // --------------------------------------------------------
    // SAFE CROSSING
    // --------------------------------------------------------

    if(agent.progress>=10) {

        safeCrossings++;

        message=
            "*** SAFE CROSSING ***";

        messageTimer=2.0;

        setTimeout(
            resetEpisode,
            2000
        );

        return;
    }


    // if car goes too far,
    // restart car but NOT pedestrian

    if(car.x>45) {

        car.x=-45;
    }
}


// ============================================================
// DRAW CAR
// ============================================================

function drawCar() {

    let p=
        projectPoint(
            car.x,
            .65,
            car.z
        );


    if(!p)
        return;


    if(
        p.depth<1 ||
        p.depth>80
    )
        return;


    let c=
        Math.round(p.col);

    let r=
        Math.round(p.row);


    let scale=
        clamp(
            30/p.depth,
            2,
            6
        );


    let s=
        Math.round(scale);


    // larger car

    if(s>=3) {

        let lines=[

            "    ______    ",
            " __/______\\__ ",
            "|  []    []  |",
            "O============O"
        ];


        for(
            let j=0;
            j<lines.length;
            j++
        ) {

            let line=
                lines[j];


            for(
                let k=0;
                k<line.length;
                k++
            ) {

                if(line[k]!==" ") {

                    setChar(

                        c-
                        Math.floor(
                            line.length/2
                        )
                        +k,

                        r-3+j,

                        line[k],

                        p.depth
                    );
                }
            }
        }

    } else {

        setChar(
            c,
            r,
            "#",
            p.depth
        );
    }
}


// ============================================================
// DRAW RL AGENT
//
// BIG @ HEAD SO IT IS IMPOSSIBLE TO MISS.
// ============================================================

function drawAgent() {

    let base=
        projectPoint(
            agent.x,
            0,
            agent.z
        );


    let head=
        projectPoint(
            agent.x,
            2.0,
            agent.z
        );


    if(
        !base ||
        !head
    )
        return;


    let c=
        Math.round(
            head.col
        );


    let r=
        Math.round(
            head.row
        );


    let depth=
        base.depth;


    const person=[

        "  @@@  ",
        " @@@@@ ",
        "   @   ",
        "  /|\\  ",
        " / | \\ ",
        "   |   ",
        "  / \\  ",
        " /   \\ "
    ];


    for(
        let j=0;
        j<person.length;
        j++
    ) {

        let line=
            person[j];


        for(
            let k=0;
            k<line.length;
            k++
        ) {

            if(line[k]!==" ") {

                setChar(

                    c-
                    Math.floor(
                        line.length/2
                    )
                    +k,

                    r+j,

                    line[k],

                    depth-.1
                );
            }
        }
    }
}


// ============================================================
// STREET LIGHTS
// ============================================================

function drawStreetLight(x,z) {

    let base=
        projectPoint(
            x,0,z
        );

    let top=
        projectPoint(
            x,4,z
        );


    if(!base || !top)
        return;


    let c=
        Math.round(top.col);

    let r1=
        Math.round(top.row);

    let r2=
        Math.round(base.row);


    setChar(
        c,
        r1,
        "*",
        base.depth
    );


    for(
        let r=r1+1;
        r<=r2;
        r++
    ) {

        setChar(
            c,
            r,
            "|",
            base.depth
        );
    }
}


// ============================================================
// RENDER
// ============================================================

function render() {

    buffer=
        new Array(
            COLS*ROWS
        );


    depthBuffer=
        new Array(
            COLS*ROWS
        );


    buffer.fill(" ");

    depthBuffer.fill(
        Infinity
    );


    // ========================================================
    // RAY CAST CITY
    // ========================================================

    for(
        let row=0;
        row<ROWS;
        row++
    ) {

        for(
            let col=0;
            col<COLS;
            col++
        ) {

            let sx=
                (
                    col/COLS-.5
                )*1.55;


            let sy=
                (
                    .5-row/ROWS
                )*.95;


            let dx=sx;
            let dy=sy;
            let dz=1;


            let len=
                Math.sqrt(
                    dx*dx+
                    dy*dy+
                    dz*dz
                );


            dx/=len;
            dy/=len;
            dz/=len;


            let hit=
                traceRay(
                    dx,dy,dz
                );


            let character=" ";

            let depth=Infinity;


            if(hit) {

                depth=hit.t;


                if(
                    hit.type===
                    "ground"
                ) {

                    character=
                        groundCharacter(
                            hit.x,
                            hit.z,
                            hit.t
                        );

                } else {

                    let light=
                        .25;


                    if(hit.ny>0)
                        light=.65;

                    if(hit.nx!==0)
                        light=.38;

                    if(hit.nz!==0)
                        light=.28;


                    light *=
                        Math.max(
                            .2,
                            1-hit.t/
                            VIEW_DISTANCE
                        );


                    character=
                        shade(light);
                }

            } else {

                character=
                    shade(.03);
            }


            let index=
                row*COLS+
                col;


            buffer[index]=
                character;


            depthBuffer[index]=
                depth;
        }
    }


    // ========================================================
    // STREET LIGHTS
    // ========================================================

    for(
        let x=-40;
        x<=40;
        x+=12
    ) {

        drawStreetLight(
            x,
            INTERSECTION_Z-8
        );

        drawStreetLight(
            x,
            INTERSECTION_Z+8
        );
    }


    // ========================================================
    // CAR + AGENT
    // ========================================================

    drawCar();

    drawAgent();


    // ========================================================
    // DRAW ASCII BUFFER
    // ========================================================

    ctx.fillStyle="#000";

    ctx.fillRect(
        0,0,W,H
    );


    ctx.fillStyle=
        "#a8c3b8";


    ctx.font=
        CELL_H+
        "px monospace";


    for(
        let row=0;
        row<ROWS;
        row++
    ) {

        let line="";


        for(
            let col=0;
            col<COLS;
            col++
        ) {

            line +=
                buffer[
                    row*COLS+
                    col
                ];
        }


        ctx.fillText(
            line,
            0,
            row*CELL_H
        );
    }


    // ========================================================
    // HUD
    // ========================================================

    ctx.font=
        "16px monospace";


    ctx.fillStyle=
        "#ffffff";


    ctx.fillText(

        "RL PEDESTRIAN EXPERIMENT",

        18,
        22
    );


    ctx.fillText(

        "[1] RANDOM    [2] TRAINED",

        18,
        48
    );


    ctx.fillText(

        "MODE: "+mode,

        18,
        76
    );


    ctx.fillText(

        "ACTION: "+
        (
            agent.action===1
            ? "WALK"
            : "WAIT"
        ),

        18,
        102
    );


    ctx.fillText(

        "SAFE CROSSINGS: "+
        safeCrossings,

        18,
        128
    );


    ctx.fillText(

        "COLLISIONS: "+
        collisions,

        18,
        154
    );


    // ========================================================
    // LABEL AGENT
    // ========================================================

    let label=
        projectPoint(
            agent.x,
            2.8,
            agent.z
        );


    if(label) {

        ctx.fillText(

            mode+
            " AGENT",

            label.col*
            CELL_W-55,

            label.row*
            CELL_H-20
        );
    }


    // ========================================================
    // BIG EVENT MESSAGE
    // ========================================================

    if(messageTimer>0) {

        ctx.font=
            "bold 34px monospace";


        ctx.fillText(

            message,

            W/2-
            message.length*10,

            80
        );
    }
}


// ============================================================
// LOOP
// ============================================================

let last=
    performance.now();


function animate(time) {

    let dt=
        Math.min(
            .05,
            (time-last)/1000
        );


    last=time;


    updateWorld(dt);

    render();


    requestAnimationFrame(
        animate
    );
}


resetEpisode();

requestAnimationFrame(
    animate
);

</script>

</body>
</html>




```





