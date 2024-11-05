const express = require("express");
const multer = require("multer");
const { Storage } = require("@google-cloud/storage");
const admin = require("firebase-admin");
const tf = require("@tensorflow/tfjs-node");
const { v4: uuidv4 } = require("uuid");
const fs = require("fs");
const path = require("path");
const cors = require("cors");

const app = express();
app.use(cors());
const PORT = process.env.PORT || 8080;
const HOST = process.env.NODE_ENV === "production" ? "0.0.0.0" : "localhost";

// Inisialisasi Firebase tanpa service account (Cloud Run akan menggunakan Workload Identity)
admin.initializeApp();
const db = admin.firestore();

const bucketName = "asclepius-khoirul";
const modelFolder = "models"; // Nama folder di dalam bucket
const localModelPath = "./temp-model"; // Folder lokal untuk menyimpan model

let model; // Untuk menyimpan model setelah dimuat
const storage = new Storage(); // Inisialisasi Google Cloud Storage

// Fungsi untuk mengunduh seluruh folder model dari Cloud Storage
// Fungsi untuk mengunduh seluruh folder model dari Cloud Storage
async function downloadModelFolder() {
  const [files] = await storage.bucket(bucketName).getFiles({ prefix: modelFolder });

  // Buat folder lokal jika belum ada
  if (!fs.existsSync(localModelPath)) {
    fs.mkdirSync(localModelPath);
  }

  await Promise.all(
    files.map((file) => {
      const localFilePath = path.join(localModelPath, path.basename(file.name)); // Simpan dengan nama file asli
      return file.download({ destination: localFilePath });
    })
  );
  console.log("Model folder downloaded successfully");
}

// Fungsi untuk memuat model dari folder lokal
async function loadModelFromFolder() {
  model = await tf.loadGraphModel(`file://${localModelPath}/model.json`); // Pastikan jalur file model.json
  console.log("Model loaded successfully");
}

// Fungsi utama untuk mengunduh dan memuat model
async function initializeModel() {
  // Buat folder lokal jika belum ada
  if (!fs.existsSync(localModelPath)) {
    fs.mkdirSync(localModelPath);
  }

  // Unduh folder model
  await downloadModelFolder();
  // Muat model dari folder lokal
  await loadModelFromFolder();
}

// Muat model pada startup
initializeModel()
  .then(() => {
    console.log("Model initialized successfully");
  })
  .catch((error) => console.error("Error initializing model:", error));

// Konfigurasi multer untuk upload file gambar
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
