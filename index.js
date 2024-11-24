const express = require("express");
const multer = require("multer");
const admin = require("firebase-admin");
const tf = require("@tensorflow/tfjs-node");
// const { Storage } = require("@google-cloud/storage");
const { v4: uuidv4 } = require("uuid");
const cors = require("cors");

// Ganti dengan nama bucket dan file yang sesuai
// const bucketName = "asclepius-khoirul";
// const fileName = "models/model.json";

// const storage = new Storage();

const app = express();
app.use(cors());
const PORT = process.env.PORT || 8080;
const HOST = process.env.NODE_ENV === "production" ? "0.0.0.0" : "localhost";

admin.initializeApp();
const db = admin.firestore();

let model;

async function loadModelFromFolder() {
  model = await tf.loadGraphModel(`https://storage.googleapis.com/asclepius-refilnovianto/submissions-model/model.json`); // Pastikan jalur file model.json
  console.log("Model loaded successfully");
}

loadModelFromFolder()
  .then(() => {
    console.log("Model initialized successfully");
  })
  .catch((error) => console.error("Error initializing model:", error));

const upload = multer({
  limits: { fileSize: 1000000 },
  fileFilter(req, file, cb) {
    if (!file.mimetype.startsWith("image/")) {
      cb(new Error("File harus berupa gambar"));
    } else {
      cb(null, true);
    }
  },
});

// Endpoint untuk prediksi
app.post("/predict", upload.single("image"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ status: "fail", message: "Tidak ada file yang diunggah" });
  }

  try {
    const imageBuffer = req.file.buffer;
    const tensor = tf.node.decodeImage(imageBuffer).resizeBilinear([224, 224]).expandDims();
    const prediction = model.predict(tensor);
    const result = prediction.dataSync()[0] > 0.5 ? "Cancer" : "Non-cancer";
    const suggestion = result === "Cancer" ? "Segera periksa ke dokter!" : "Penyakit kanker tidak terdeteksi.";

    const id = uuidv4();
    const createdAt = new Date().toISOString();
    const responseData = { id, result, suggestion, createdAt };

    // Simpan hasil prediksi ke Firestore
    await db.collection("predictions").doc(id).set(responseData);

    res.status(201).json({
      status: "success",
      message: "Model is predicted successfully",
      data: responseData,
    });
  } catch (error) {
    res.status(400).json({ status: "fail", message: "Terjadi kesalahan dalam melakukan prediksi" });
  }
});

// Endpoint untuk mengambil riwayat prediksi
app.get("/predict/histories", async (req, res) => {
  try {
    const snapshot = await db.collection("predictions").get();
    const histories = snapshot.docs.map((doc) => ({
      id: doc.id,
      history: doc.data(),
    }));

    res.status(200).json({ status: "success", data: histories });
  } catch (error) {
    res.status(500).json({ status: "fail", message: "Gagal mengambil data riwayat prediksi" });
  }
});

// Error handling untuk Multer
app.use((error, req, res, next) => {
  if (error instanceof multer.MulterError && error.code === "LIMIT_FILE_SIZE") {
    return res.status(413).json({
      status: "fail",
      message: "Payload content length greater than maximum allowed: 1000000",
    });
  }
  // res.status(400).json({ status: "fail", message: error.message });
});

// Jalankan server
app.listen(PORT, HOST, () => {
  console.log(`Server is running on ${HOST}/${PORT}`);
});
