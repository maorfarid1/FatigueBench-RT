const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch(); const p = await b.newPage({viewport:{width:2016,height:848}, deviceScaleFactor:2});
  await p.goto('file://' + process.argv[2], {waitUntil:'networkidle'}); await p.waitForTimeout(1500);
  await p.pdf({path: process.argv[3], width:'2016px', height:'848px', printBackground:true, pageRanges:'1'});
  await p.screenshot({path: process.argv[4]});
  await b.close();
})();
