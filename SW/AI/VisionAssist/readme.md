<div align="center">
  
# VisionAssist
**Empowering Spatial Awareness and Indoor Navigation through On-Device AI**

</div>

---

## About VisionAssist
**VisionAssist** is a native mobile application meticulously engineered to facilitate safe, independent, and seamless indoor navigation for individuals with visual impairments and complete blindness. By leveraging highly compressed, on-device artificial intelligence models, the application interprets the visual world into accessible audio and high-contrast visual contexts. 

---

## The Problem it Solves
Globally, over 2.2 billion people are affected by varying degrees of visual impairment. 
* **Indoor Vulnerability:** Statistics show that a staggering **78% of assistance requests** from visually impaired individuals happen in indoor spaces.
* **High Costs:** Traditional solutions often require significant financial investments in expensive, specialized optical hardware and devices.

## 💡 How it Solves It
VisionAssist acts as a digital eye, replacing expensive hardware with a purely software-driven approach designed to make indoor navigation easier. 
* **AI Object Tracking:** The app identifies objects, calculating both their exact locations (bounding boxes) and their classification.
* **Multithreaded Architecture:** It uses advanced multithreading to manage multiple AI models simultaneously, providing context through either visual or audio cues based on the user's specific needs.

---

## ✚ Stand-Out Features (The Competitive Edge)
Unlike standard accessibility tools, VisionAssist incorporates highly specialized engineering mechanisms to maximize speed and privacy:

* **100% Offline Capability:** Complete independence from internet access. All AI models (YOLOv8, BLIP, Vosk) are compressed inside the APK and execute directly on the user's mobile hardware.
* **FindMyObject Feature:** A specialized utility built from the ground up to help users locate specific lost or misplaced items in their environment.
* **Algorithmic Acceleration:** 
  * **YOLOv8 Optimization:** Utilizes custom `Detector Pooling` and `Speed Over Accuracy (SoA)` algorithms to process continuous frames with minimal latency.
  * **BLIP Optimization:** Implements `Hash Caching` to dramatically speed up image captioning activities by preventing redundant AI processing.
* **Continuous Frame Processing:** Provides a true "Live" experience rather than relying solely on static, button-triggered image captures.

---

## Core Facilities & Capabilities
* **Dual User Targeting:** Deeply customized UI/UX catering specifically to two distinct groups: *Low Vision Users* and *Completely Blind Users*.
* **Auto Illumination:** Automatically detects low-light environments and manages the camera flash to ensure high precision in AI responses.
* **QuickAction:** Allows users to bypass menus and jump directly into detection through fast, accessible shortcuts.
* **Bilingual Interface:** Supports multiple languages for wider accessibility.
* **Hybrid Data Persistence:** Offers local storage for absolute privacy, alongside remote synchronization via Firebase for cross-device profile management.

---

## App Description & User Experience

Upon launching VisionAssist, users are guided through a personalized profile creation where they set their language, select their specific impairment profile, and choose between local or Firebase remote synchronization.

###  Profile: _Low Eyesight_
Designed for users with partial sight, maximizing visual contrast and readability.
* **Customization:** Users can set custom colors for detection results, adjust text/bounding box sizes, and enable haptic vibration feedback.
* **Static & Live Detection:** `StaticDetectionActivity` processes single photos, while `LiveDetectionActivity` processes continuous frames using Semaphores and multithreading.
* **FindMyObject:** Monitors battery temperature/light and displays bounding boxes immediately when the target object is found.
* **Reports:** Access to `EnvironmentReportsActivity` to view the most common scenes, objects, and average AI response times.

###  Profile: _Blindness_
A radical UX shift designed strictly for auditory navigation and touch accessibility.
* **Customization:** Users can fine-tune the Text-to-Speech (TTS) engine's pitch and speed. 
* **Audio-First Navigation:** Screen navigation is driven by physical volume buttons and highly interactive surfaces that respond even if the user taps outside a designated button. Includes a toggle mechanism between standard TTS and TalkBack.
* **BlindFindMyObject:** Instead of visual boxes, the app generates detailed sentences describing the object's class and exact position in the room, spoken aloud by the TTS.
* **BlindDetection:** Runs continuous `Detector Pooling` and `SoA` until an object is found, immediately generating and speaking a descriptive sentence.
* **BlindCaption:** Similar to standard captioning, but the generated scene description is spoken out loud.