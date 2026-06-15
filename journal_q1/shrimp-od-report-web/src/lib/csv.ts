import Papa from "papaparse";
export type Row = Record<string, string>;
export async function loadCsv(path: string): Promise<Row[]> {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  const text = await response.text();
  return Papa.parse<Row>(text, { header: true, skipEmptyLines: true }).data;
}
export async function loadText(path: string): Promise<string> {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.text();
}
