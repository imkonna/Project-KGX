// Layout bounds check: reports any element overflowing the A0 page (badCount must be 0).
const puppeteer = require("puppeteer-core");
const path = require("path");
const CHROME = "C:/Program Files/Google/Chrome/Application/chrome.exe";
const fileUrl = "file:///" + path.join(__dirname, "Poster_GDL_Drug_Toxicity_A0.html").replace(/\\/g, "/");
(async () => {
  const b = await puppeteer.launch({ executablePath: CHROME, headless: "new", args: ["--allow-file-access-from-files"] });
  const p = await b.newPage();
  await p.setViewport({ width: 1190, height: 1683, deviceScaleFactor: 1 });
  await p.goto(fileUrl, { waitUntil: "networkidle0" });
  const res = await p.evaluate(() => {
    const pg = document.querySelector(".page").getBoundingClientRect();
    const bad = [];
    document.querySelectorAll(".page *").forEach((el) => {
      const r = el.getBoundingClientRect();
      if (r.width === 0 || r.height === 0) return;
      if (getComputedStyle(el).position === "absolute") return;
      if (r.right > pg.right + 1.5 || r.left < pg.left - 1.5 || r.bottom > pg.bottom + 1.5)
        bad.push((el.className || el.tagName) + " R" + Math.round(r.right) + " B" + Math.round(r.bottom));
    });
    return { badCount: bad.length, bad: bad.slice(0, 12) };
  });
  console.log("badCount=" + res.badCount);
  res.bad.forEach((x) => console.log("  " + x));
  await b.close();
})();
