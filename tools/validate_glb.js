// node tools/validate_glb.js web/models/*.glb   (needs: npm i -g gltf-validator)
const fs = require('fs');
const path = require('path');
const validator = require(path.join(require('child_process').execSync('npm root -g').toString().trim(), 'gltf-validator'));
(async () => {
  for (const f of process.argv.slice(2)) {
    const report = await validator.validateBytes(new Uint8Array(fs.readFileSync(f)), { maxIssues: 20 });
    const { numErrors, numWarnings, numInfos, numHints } = report.issues;
    console.log(`${f}: errors=${numErrors} warnings=${numWarnings} infos=${numInfos} hints=${numHints}`);
    for (const m of report.issues.messages.slice(0, 12)) console.log(`   [${m.severity}] ${m.code}: ${m.message} @ ${m.pointer}`);
  }
})();
