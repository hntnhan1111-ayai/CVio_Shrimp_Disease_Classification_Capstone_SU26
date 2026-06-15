import type { Row } from "./csv";
export const number = (value?: string) => value === undefined || value === "" || !Number.isFinite(Number(value)) ? null : Number(value);
export const fmt = (value: unknown, digits = 3) => value === null || value === undefined || value === "" || !Number.isFinite(Number(value)) ? "N/A" : Number(value).toFixed(digits);
export const metric = (rows: Row[], key: string, mode: "max"|"min" = "max") => rows.filter(r => r.status === "ok" && number(r[key]) !== null).sort((a,b) => mode === "max" ? Number(b[key])-Number(a[key]) : Number(a[key])-Number(b[key]))[0];
export const sum = (rows: Row[], key: string) => rows.reduce((total,row) => total + (number(row[key]) ?? 0), 0);
