const localVideo = document.getElementById("localVideo");
const chatInput = document.getElementById("chatInput");
const sendText = document.getElementById("sendText");

let websocket = new WebSocket("ws://localhost:8000/webrtc");

websocket.onopen = () => {
  console.log("WebSocket connection established");
};

websocket.onclose = () => {
  console.log("WebSocket connection closed");
};

websocket.onerror = (error) => {
  console.error("WebSocket error:", error);
};

// Capture video and audio
navigator.mediaDevices
  .getUserMedia({ video: true, audio: true })
  .then((stream) => {
    localVideo.srcObject = stream;

    const mediaRecorder = new MediaRecorder(stream, { mimeType: "video/webm" });

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        websocket.send(event.data);
      }
    };

    mediaRecorder.start(1000); // Send chunks every second
  })
  .catch((error) => {
    console.error("Error accessing media devices:", error);
  });

// Send text messages
sendText.addEventListener("click", () => {
  const text = chatInput.value.trim();
  if (text) {
    websocket.send(text);
    chatInput.value = "";
  }
});
