# LightEndoStereo

Real-time acquisition of accurate depth of scene is essential for automated robotic minimally invasive surgery, and stereo matching with binocular endoscopy can generate such depth. However, existing algorithms struggle with ambiguous tissue boundaries and real-time performance in prevalent high-resolution endoscopic scenes. We propose LightEndoStereo, a lightweight real-time stereo matching method for endoscopic images. We introduce a 3D Mamba Coordinate Attention module to streamline the cost aggregation process by generating position-sensitive attention maps and capturing long-range dependencies across spatial dimensions using the Mamba block. Additionally, we introduce a High-Frequency Disparity Optimization module to refine disparity estimates at tissue boundaries by enhancing high-frequency information in the wavelet domain. Our method is evaluated on the SCARED and SERV-CT datasets, achieving state-of-the-art matching accuracy and a real-time inference speed of <span style="color: blue;">42 FPS</span>.

## Framework
![framework](./assets/framework.png)


## Samples
Using dataset [SCARED](https://endovissub2019-scared.grand-challenge.org) and [SERV-CT](https://www.ucl.ac.uk/interventional-surgical-sciences/weiss-open-research/weiss-open-data-server/serv-ct).
### Samples on SCARED
![sample](./assets/sample1.png)


<video id="myVideo" src="assets/videos/LightEndoStereo_demo.mp4" controls></video>
<button onclick="playVideo()">Play</button>
<button onclick="pauseVideo()">Pause</button>
<script>
  function playVideo() {
    document.getElementById('myVideo').play();
  }
  function pauseVideo() {
    document.getElementById('myVideo').pause();
  }
</script>

### Samples on SERV-CT
![sample](./assets/sample2.png)

