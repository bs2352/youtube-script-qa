export function hms2s (hms: string) {
    const [h, m, s] = hms.split(':').map(Number)
    return h * 3600 + m * 60 + s
}

export function s2hms (s: number) {
    const sec: number = Math.round(s);
    const h = Math.floor(sec / 3600).toString().padStart(2, '0');
    const m = Math.floor((sec % 3600) / 60).toString().padStart(2, '0');
    const ss = ((sec % 3600) % 60).toString().padStart(2, '0');
    return `${h}:${m}:${ss}`
}