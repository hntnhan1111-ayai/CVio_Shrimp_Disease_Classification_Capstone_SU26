import type { ReactNode } from "react";
export function WarningBox({children,tone="warning"}:{children:ReactNode;tone?:"warning"|"info"|"success"}) { return <div className={`notice ${tone}`}>{children}</div>; }
