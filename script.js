import {
    FilesetResolver,
    HandLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";


const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const startButton = document.getElementById("startButton");
const gestureText = document.getElementById("gesture");

const reactionImage = document.getElementById("reactionImage");
const reactionText = document.getElementById("reactionText");

let handLandmarker = null;
let running = false;


/* ==========================================
   CREATE MEDIAPIPE HAND LANDMARKER
========================================== */

async function createHandLandmarker() {

    gestureText.textContent = "Loading MediaPipe...";

    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm"
    );

    handLandmarker =
        await HandLandmarker.createFromOptions(
            vision,
            {
                baseOptions: {
                    modelAssetPath:
                        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                },

                runningMode: "VIDEO",

                numHands: 2,

                minHandDetectionConfidence: 0.5,

                minHandPresenceConfidence: 0.5,

                minTrackingConfidence: 0.5
            }
        );

}


/* ==========================================
   START CAMERA
========================================== */

async function startCamera() {

    try {

        startButton.disabled = true;

        startButton.textContent = "Loading...";

        await createHandLandmarker();


        const stream =
            await navigator.mediaDevices.getUserMedia({
                video: {
                    width: 640,
                    height: 480
                },
                audio: false
            });


        video.srcObject = stream;

        running = true;

        startButton.textContent = "Camera Running";

        gestureText.textContent = "No gesture";


        video.addEventListener(
            "loadeddata",
            () => {

                predict();

            },
            { once: true }
        );


    } catch (error) {

        console.error(error);

        gestureText.textContent =
            "Camera error";

        startButton.disabled = false;

        startButton.textContent =
            "Start Camera";

        alert(
            "Could not start the camera. Check the browser permissions and Console."
        );

    }

}


/* ==========================================
   FINGER IS UP
========================================== */

function fingerIsUp(
    hand,
    tip,
    pip
) {

    return hand[tip].y < hand[pip].y;

}


/* ==========================================
   COUNT FINGERS
========================================== */

function countFingers(hand) {

    let fingers = 0;

    if (fingerIsUp(hand, 8, 6)) {
        fingers++;
    }

    if (fingerIsUp(hand, 12, 10)) {
        fingers++;
    }

    if (fingerIsUp(hand, 16, 14)) {
        fingers++;
    }

    if (fingerIsUp(hand, 20, 18)) {
        fingers++;
    }

    return fingers;

}


/* ==========================================
   PEACE
========================================== */

function isPeace(hand) {

    const indexUp =
        fingerIsUp(hand, 8, 6);

    const middleUp =
        fingerIsUp(hand, 12, 10);

    const ringDown =
        !fingerIsUp(hand, 16, 14);

    const pinkyDown =
        !fingerIsUp(hand, 20, 18);


    return (
        indexUp &&
        middleUp &&
        ringDown &&
        pinkyDown
    );

}


/* ==========================================
   SHH
========================================== */

function isShutUp(hand) {

    const indexUp =
        fingerIsUp(hand, 8, 6);

    const middleDown =
        !fingerIsUp(hand, 12, 10);

    const ringDown =
        !fingerIsUp(hand, 16, 14);

    const pinkyDown =
        !fingerIsUp(hand, 20, 18);


    return (
        indexUp &&
        middleDown &&
        ringDown &&
        pinkyDown
    );

}


/* ==========================================
   SHOW REACTION
========================================== */

function showReaction(
    gesture,
    image
) {

    gestureText.textContent =
        gesture;

    reactionImage.src =
        image;

    reactionImage.style.display =
        "block";

    reactionText.style.display =
        "none";

}


/* ==========================================
   CLEAR REACTION
========================================== */

function clearReaction() {

    gestureText.textContent =
        "No gesture";

    reactionImage.style.display =
        "none";

    reactionText.style.display =
        "block";

    reactionText.textContent =
        "Make a gesture";

}


/* ==========================================
   DRAW HAND
========================================== */

function drawHand(hand) {

    const connections = [

        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],

        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],

        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],

        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],

        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],

        [5, 9],
        [9, 13],
        [13, 17]

    ];


    ctx.fillStyle = "lime";

    ctx.strokeStyle = "lime";

    ctx.lineWidth = 2;


    for (const landmark of hand) {

        const x =
            landmark.x * canvas.width;

        const y =
            landmark.y * canvas.height;


        ctx.beginPath();

        ctx.arc(
            x,
            y,
            5,
            0,
            Math.PI * 2
        );

        ctx.fill();

    }


    for (const [start, end] of connections) {

        const x1 =
            hand[start].x * canvas.width;

        const y1 =
            hand[start].y * canvas.height;

        const x2 =
            hand[end].x * canvas.width;

        const y2 =
            hand[end].y * canvas.height;


        ctx.beginPath();

        ctx.moveTo(x1, y1);

        ctx.lineTo(x2, y2);

        ctx.stroke();

    }

}


/* ==========================================
   DETECT HANDS
========================================== */

function predict() {

    if (!running) {
        return;
    }


    if (video.readyState < 2) {

        requestAnimationFrame(predict);

        return;

    }


    canvas.width =
        video.videoWidth;

    canvas.height =
        video.videoHeight;


    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );


    const results =
        handLandmarker.detectForVideo(
            video,
            performance.now()
        );


    clearReaction();


    if (
        results.landmarks &&
        results.landmarks.length > 0
    ) {


        /* DRAW HANDS */

        for (
            const hand
            of results.landmarks
        ) {

            drawHand(hand);

        }


        const numberOfHands =
            results.landmarks.length;


        /* ==================================
           TWO HANDS UP
        ================================== */

        if (numberOfHands === 2) {

            const fingers1 =
                countFingers(
                    results.landmarks[0]
                );

            const fingers2 =
                countFingers(
                    results.landmarks[1]
                );


            if (
                fingers1 >= 3 &&
                fingers2 >= 3
            ) {

                showReaction(
                    "HANDS UP",
                    "gestures/hands%20up.jpg"
                );

            }

        }


        /* ==================================
           ONE HAND
        ================================== */

        else if (numberOfHands === 1) {

            const hand =
                results.landmarks[0];


            if (isPeace(hand)) {

                showReaction(
                    "PEACE",
                    "gestures/peace.jpg"
                );

            }

            else if (isShutUp(hand)) {

                showReaction(
                    "SHH!",
                    "gestures/shut%20up.jpg"
                );

            }

        }

    }


    requestAnimationFrame(predict);

}


/* ==========================================
   BUTTON
========================================== */

startButton.addEventListener(
    "click",
    startCamera
);