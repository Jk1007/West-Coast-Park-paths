const { Client } = require('pg');
require('dotenv').config();

async function checkSchema() {
  const client = new Client({
    connectionString: process.env.DATABASE_URL
  });
  try {
    await client.connect();
    const res = await client.query(`
      SELECT column_name, data_type 
      FROM information_schema.columns 
      WHERE table_name = 'incidents';
    `);
    console.log("Columns in 'incidents' table:");
    console.log(res.rows);
  } catch (err) {
    console.error("Error checking schema:", err);
  } finally {
    await client.end();
  }
}

checkSchema();
