// Optional test-only engine. Never reads DATABASE_URL or opens a network connection.
// API/extension reference: https://pglite.dev/extensions/
const { createRequire } = require("node:module");
const { join } = require("node:path");
const { createInterface } = require("node:readline");
const testRequire = createRequire(join(process.env.PGLITE_DEPS_DIR || __dirname, "package.json"));
const { PGlite } = testRequire("@electric-sql/pglite");
const { vector } = testRequire("@electric-sql/pglite-pgvector");

(async () => {
  const pg = await PGlite.create({ extensions: { vector } });
  await pg.exec("CREATE EXTENSION vector");
  process.stdout.write(JSON.stringify({ ready: true }) + "\n");
  const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
  for await (const line of input) {
    try {
      const { sql, params, exec } = JSON.parse(line);
      const result = exec ? await pg.exec(sql) : await pg.query(sql, params || []);
      process.stdout.write(JSON.stringify({ result }) + "\n");
    } catch (error) {
      process.stdout.write(JSON.stringify({ error: error.message }) + "\n");
    }
  }
  await pg.close();
})().catch(error => { process.stderr.write(error.message); process.exitCode = 1; });
