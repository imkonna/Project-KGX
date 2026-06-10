const fs = require("fs");
const path = require("path");
const puppeteer = require("puppeteer-core");

const ROOT = path.resolve(__dirname, "..");
const POSTER_DIR = path.join(ROOT, "poster");
const COMPILED_DIR = path.join(POSTER_DIR, "compiled");
const CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe";

const htmlPath = path.join(POSTER_DIR, "Poster_GDL_Drug_Toxicity_A0.html");
const previewPath = path.join(POSTER_DIR, "Poster_GDL_Drug_Toxicity_A0_preview.png");
const activePdfPath = path.join(POSTER_DIR, "Poster_GDL_Drug_Toxicity_A0.pdf");
const compiledHtmlPath = path.join(COMPILED_DIR, "final_clean.html");
const compiledPreviewPath = path.join(COMPILED_DIR, "final_clean_preview.png");
const compiledHighresJpgPath = path.join(COMPILED_DIR, "final_clean_highres.jpg");
const compiledHighresPdfPath = path.join(COMPILED_DIR, "final_clean_highres.pdf");

function fileUrl(filePath) {
  return "file:///" + filePath.replace(/\\/g, "/");
}

async function renderHtmlPreview(browser) {
  const page = await browser.newPage();
  await page.setViewport({ width: 1190, height: 1683, deviceScaleFactor: 1 });
  await page.goto(fileUrl(htmlPath), { waitUntil: "networkidle0" });
  await page.screenshot({ path: previewPath, fullPage: true });
  await page.screenshot({ path: compiledPreviewPath, fullPage: true });
  fs.copyFileSync(htmlPath, compiledHtmlPath);
  await page.close();
}

async function renderHighresJpg(browser) {
  const page = await browser.newPage();
  await page.setViewport({ width: 1190, height: 1683, deviceScaleFactor: 2 });
  await page.goto(fileUrl(htmlPath), { waitUntil: "networkidle0" });
  await page.screenshot({
    path: compiledHighresJpgPath,
    fullPage: true,
    type: "jpeg",
    quality: 88,
  });
  await page.close();
}

async function renderStablePdf(browser) {
  const data = fs.readFileSync(compiledHighresJpgPath).toString("base64");
  const page = await browser.newPage();
  await page.setViewport({ width: 1190, height: 1683, deviceScaleFactor: 1 });
  await page.setContent(
    `<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    @page { size: 841mm 1189mm; margin: 0; }
    html, body { margin: 0; padding: 0; width: 841mm; height: 1189mm; overflow: hidden; background: white; }
    img { display: block; width: 841mm; height: 1189mm; object-fit: fill; }
  </style>
</head>
<body>
  <img id="poster" src="data:image/jpeg;base64,${data}">
</body>
</html>`,
    { waitUntil: "load" }
  );
  await page.waitForFunction(
    () => document.getElementById("poster")?.complete && document.getElementById("poster")?.naturalWidth > 0
  );
  const pdfOptions = {
    width: "841mm",
    height: "1189mm",
    printBackground: true,
    margin: { top: "0mm", right: "0mm", bottom: "0mm", left: "0mm" },
  };
  let renderedActivePdf = activePdfPath;
  try {
    await page.pdf({ ...pdfOptions, path: activePdfPath });
  } catch (error) {
    if (error.code !== "EBUSY") {
      throw error;
    }
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    renderedActivePdf = path.join(COMPILED_DIR, `Poster_GDL_Drug_Toxicity_A0_${stamp}.pdf`);
    await page.pdf({ ...pdfOptions, path: renderedActivePdf });
    console.warn(`Active PDF was locked. Wrote fallback: ${renderedActivePdf}`);
  }
  try {
    await page.pdf({ ...pdfOptions, path: compiledHighresPdfPath });
  } catch (error) {
    if (error.code !== "EBUSY") {
      throw error;
    }
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    const fallbackPath = path.join(COMPILED_DIR, `final_clean_highres_${stamp}.pdf`);
    await page.pdf({ ...pdfOptions, path: fallbackPath });
    console.warn(`Compiled PDF was locked. Wrote fallback: ${fallbackPath}`);
  }
  await page.close();
  return renderedActivePdf;
}

(async () => {
  fs.mkdirSync(COMPILED_DIR, { recursive: true });
  const browser = await puppeteer.launch({
    executablePath: CHROME,
    headless: "new",
    args: ["--allow-file-access-from-files", "--disable-web-security"],
  });

  await renderHtmlPreview(browser);
  await renderHighresJpg(browser);
  const renderedPdfPath = await renderStablePdf(browser);
  await browser.close();

  const mb = fs.statSync(renderedPdfPath).size / 1024 / 1024;
  console.log(`Rendered poster PDF: ${renderedPdfPath} (${mb.toFixed(2)} MB)`);
})();
