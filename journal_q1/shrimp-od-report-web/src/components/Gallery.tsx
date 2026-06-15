import { useState } from "react";
import { gallery } from "../lib/paths";
import { ImageFigure } from "./ImageFigure";
const pages={BG:5,WSSV:7,WSSV_BG:5};
export function Gallery(){const [tab,setTab]=useState<keyof typeof pages>("BG"); const files=[`overlay_samples_${tab}.jpg`,...Array.from({length:pages[tab]},(_,i)=>`FULL_overlay_${tab}_page_${String(i+1).padStart(3,"0")}.jpg`)];return <><div className="tabs">{Object.keys(pages).map(k=><button className={tab===k?"active":""} onClick={()=>setTab(k as keyof typeof pages)} key={k}>{k}</button>)}</div><div className="gallery">{files.map(f=><ImageFigure key={f} src={gallery(f)} caption={f.replaceAll("_"," ").replace(".jpg","")}/>)}</div></>}
