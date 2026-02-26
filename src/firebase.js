import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";

const firebaseConfig = {
  apiKey: "AIzaSyDojs7ZIFu4EArSY3Hk7vimztgVxtzjwPE",
  authDomain: "crowdshieldweb-bf341.firebaseapp.com",
  projectId: "crowdshieldweb-bf341",
  storageBucket: "crowdshieldweb-bf341.firebasestorage.app",
  messagingSenderId: "899939472025",
  appId: "1:899939472025:web:00d29c88a4110aa4369e08",
  measurementId: "G-XQ7LP44B03"
};

let app = null;
let db = null;

try {
    if (Object.keys(firebaseConfig).length > 0) {
        app = initializeApp(firebaseConfig);
        db = getFirestore(app);
    } else {
        console.warn("Firebase config is empty. Please paste your config into src/firebase.js.");
    }
} catch (e) {
    console.error("Firebase initialization failed:", e);
}

export { app, db };
