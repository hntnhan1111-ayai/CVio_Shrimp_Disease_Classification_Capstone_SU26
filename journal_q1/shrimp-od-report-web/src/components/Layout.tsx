import type { ReactNode } from "react";
const links=["summary","dataset","labels","boxes","gallery","benchmark","failures","interpretation","recommendations","limitations"];
export function Layout({children}:{children:ReactNode}) { return <><header className="topbar"><a className="brand" href="#summary">SHRIMP OD <span>RESEARCH REPORT</span></a><nav>{links.map(x=><a key={x} href={`#${x}`}>{x}</a>)}</nav></header><main>{children}</main><footer>Static research dashboard · ShrimpDiseaseImageBD · 30-epoch screening protocol</footer></>; }
