const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch(); const p = await b.newPage({viewport:{width:1880,height:576}, deviceScaleFactor:2});
  await p.goto('file:///home/claude/paper/code/fig1v2/fig1v4.html'); await p.waitForTimeout(500);
  await p.screenshot({path:'/home/claude/paper/figs/fig1_framework.png', clip:{x:0,y:0,width:1880,height:576}});
  await p.pdf({path:'/home/claude/paper/tex/figs/fig1_framework.pdf', width:'1880px', height:'576px', printBackground:true, pageRanges:'1'});
  await b.close();
})();
