require("dotenv").config();
const express = require("express");
const fileUpload = require("express-fileupload");
const cloudinary = require("cloudinary").v2;
const { exec } = require("child_process");
const fs = require("fs");
const path = require("path");
const { v4: uuidv4 } = require("uuid");

const app = express();
const port = 3000;

// Middleware
app.use(fileUpload());

// Cloudinary config
cloudinary.config({
  cloud_name: process.env.CLOUD_NAME,
  api_key: process.env.API_KEY,
  api_secret: process.env.API_SECRET,
});

// API Endpoint
app.post("/verify", async (req, res) => {
  try {
    const file = req.files?.file;

    if (!file) {
      return res.status(400).json({ error: "No file uploaded" });
    }

    if (!fs.existsSync("./uploads")) {
      fs.mkdirSync("./uploads");
    }

    const tempInputPath = `./uploads/temp_${uuidv4()}.jpg`;
    await file.mv(tempInputPath);

    const referenceImageUrl =
      "C:/Users/Aryan Ghatge/OneDrive/Pictures/Camera Roll/Doc.jpg";

    const command = `python verify_face.py "${tempInputPath}" "${referenceImageUrl}"`;

    exec(command, (error, stdout, stderr) => {
      console.log("❌ Python Script Error:");
      console.log("Command:", command);
      console.log("STDERR:", stderr);
      console.log("STDOUT:", stdout);

      // Always cleanup uploaded image
      if (fs.existsSync(tempInputPath)) fs.unlinkSync(tempInputPath);

      if (error) {
        return res.status(500).json({
          error: "Face verification failed",
          details: stderr || error.message,
        });
      }

      try {
        const cleanOutput = stdout.trim().split("\n").pop();
        const result = JSON.parse(cleanOutput);
        res.json(result);
      } catch (parseError) {
        return res.status(500).json({
          error: "Error parsing Python output",
          details: parseError.message,
        });
      }
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.listen(port, () => {
  console.log(`✅ Server running on http://localhost:${port}`);
});
