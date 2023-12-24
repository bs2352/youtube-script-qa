export function hms2s (hms: string) {
  const [h, m, s] = hms.split(':').map(Number)
  return h * 3600 + m * 60 + s
}