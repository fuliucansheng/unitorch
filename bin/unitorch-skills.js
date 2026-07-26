#!/usr/bin/env node

const path = require("path");
const { spawnSync } = require("child_process");

const args = process.argv.slice(2);

if (args.length === 0 || args.includes("--help") || args.includes("-h")) {
  console.log(`Usage:
  unitorch install all --folder .skills --force true
  unitorch export all --folder ./agent-skills
  unitorch validate --folder .skills

This wrapper invokes:
  python3 -m unitorch.cli.copilots.skills <args>
`);
  process.exit(args.length === 0 ? 1 : 0);
}

const packageRoot = path.resolve(__dirname, "..");
const packageSrc = path.join(packageRoot, "src");
const env = { ...process.env };
env.PYTHONPATH = env.PYTHONPATH
  ? `${packageSrc}${path.delimiter}${env.PYTHONPATH}`
  : packageSrc;

const pythonCandidates = [];
if (env.PYTHON) {
  pythonCandidates.push(env.PYTHON);
}
pythonCandidates.push("python3", "python");

let lastError = null;
for (const python of [...new Set(pythonCandidates)]) {
  const result = spawnSync(
    python,
    ["-m", "unitorch.cli.copilots.skills", ...args],
    {
      stdio: "inherit",
      env,
    },
  );

  if (result.error && result.error.code === "ENOENT") {
    lastError = result.error;
    continue;
  }
  if (result.error) {
    console.error(result.error.message);
    process.exit(1);
  }
  process.exit(result.status === null ? 1 : result.status);
}

console.error(
  `Unable to find a Python interpreter. Tried: ${[
    ...new Set(pythonCandidates),
  ].join(", ")}`,
);
if (lastError) {
  console.error(lastError.message);
}
process.exit(1);
