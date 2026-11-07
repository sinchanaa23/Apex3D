// A Stanford Cars label such as "Aston Martin V8 Vantage Coupe 2012" holds the manufacturer, the model, the body style and the year.
// The backend sends these as separate fields. This reads them from the label when a response doesn't, so the page can always show all three.

export interface VehicleFacts {
  make: string
  model: string
  year: number | null
  bodyStyle: string | null
}

const MULTI_WORD_MAKES = ['AM General', 'Aston Martin', 'Land Rover', 'Mercedes-Benz', 'Rolls-Royce']
// Two-word styles first, so "Crew Cab" is not read as "Cab".
const BODY_STYLES = ['Extended Cab', 'Regular Cab', 'Crew Cab', 'Quad Cab', 'Club Cab', 'SuperCab', 'Super Cab', 'Convertible', 'Hatchback', 'Minivan', 'Sedan', 'Coupe', 'Wagon', 'SUV', 'Van']

export function describeVehicle(label: string, given?: { make?: string | null; model?: string | null; year?: number | null; body_style?: string | null }): VehicleFacts {
  const words = label.trim().split(/\s+/)
  const last = Number(words[words.length - 1])
  const hasYear = Number.isInteger(last) && last > 1900 && last < 2100
  const rest = hasYear ? words.slice(0, -1) : words

  const text = rest.join(' ')
  const make = MULTI_WORD_MAKES.find((m) => text.toLowerCase().startsWith(`${m.toLowerCase()} `)) ?? rest[0] ?? ''
  const afterMake = text.slice(make.length).trim()
  const style = BODY_STYLES.find((s) => afterMake.toLowerCase().endsWith(` ${s.toLowerCase()}`) || afterMake.toLowerCase() === s.toLowerCase())
  const model = style ? afterMake.slice(0, afterMake.length - style.length).trim() : afterMake

  return {
    make: given?.make || make,
    model: given?.model || model || afterMake,
    year: given?.year ?? (hasYear ? last : null),
    bodyStyle: given?.body_style || style || null,
  }
}
