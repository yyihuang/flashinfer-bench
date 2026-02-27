import { Definition } from "@/lib/schemas"

const TAG_STYLES: Record<string, string> = {
  status:  "bg-green-100 text-green-800 border-green-200",
  model:   "bg-blue-100 text-blue-800 border-blue-200",
  fi_api:  "bg-purple-100 text-purple-800 border-purple-200",
  tp:      "bg-slate-100 text-slate-700 border-slate-200",
  ep:      "bg-slate-100 text-slate-700 border-slate-200",
  stage:   "bg-cyan-100 text-cyan-800 border-cyan-200",
  sparse:  "bg-orange-100 text-orange-800 border-orange-200",
}

function getTagStyle(tag: string): string {
  const prefix = tag.split(":")[0]
  if (prefix === "status" && tag.includes("draft")) {
    return "bg-amber-100 text-amber-800 border-amber-200"
  }
  return TAG_STYLES[prefix] ?? "bg-gray-100 text-gray-700 border-gray-200"
}

export function DefinitionTags({ definition }: { definition: Definition }) {
  const tags = definition.tags
  if (!tags || tags.length === 0) return null

  return (
    <div className="flex flex-wrap gap-1.5">
      {tags.map((tag) => {
        const colonIdx = tag.indexOf(":")
        const prefix = colonIdx !== -1 ? tag.slice(0, colonIdx) : tag
        const value  = colonIdx !== -1 ? tag.slice(colonIdx + 1) : ""
        return (
          <span
            key={tag}
            className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium ${getTagStyle(tag)}`}
          >
            {colonIdx !== -1 ? (
              <>
                <span className="opacity-60">{prefix}:</span>
                <span>{value}</span>
              </>
            ) : (
              <span>{tag}</span>
            )}
          </span>
        )
      })}
    </div>
  )
}
