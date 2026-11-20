import { NextRequest, NextResponse } from "next/server";
import { undistortImage, BrownConradyParams } from "@/lib/brownConrady";
import sharp from "sharp";

export const maxDuration = 30;

/**
 * POST /api/correct
 * Body: FormData  { image: File, coefficients: JSON string }
 * Returns: { correctedDataUrl, metrics }
 */
export async function POST(req: NextRequest) {
  try {
    const form    = await req.formData();
    const file    = form.get("image") as File | null;
    const coefRaw = form.get("coefficients") as string | null;

    if (!file)    return NextResponse.json({ error: "No image uploaded" }, { status: 400 });
    if (!coefRaw) return NextResponse.json({ error: "No coefficients"  }, { status: 400 });

    const p: BrownConradyParams = JSON.parse(coefRaw);

    // Decode → raw RGBA
    const ab = await file.arrayBuffer();
    const { data, info } = await sharp(Buffer.from(ab))
      .ensureAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });

    const { width, height } = info;

    if (width * height > 10_000_000) {
      return NextResponse.json(
        { error: "Image too large. Max ~10 MP." },
        { status: 413 },
      );
    }

    // Undistort
    const { dst, metrics } = undistortImage(
      new Uint8ClampedArray(data.buffer),
      width, height, p,
    );

    // Encode to PNG
    const pngBuf = await sharp(Buffer.from(dst.buffer), {
      raw: { width, height, channels: 4 },
    }).png().toBuffer();

    const correctedDataUrl = `data:image/png;base64,${pngBuf.toString("base64")}`;
    return NextResponse.json({ correctedDataUrl, metrics });

  } catch (err: unknown) {
    console.error("[/api/correct]", err);
    return NextResponse.json(
      { error: err instanceof Error ? err.message : "Internal error" },
      { status: 500 },
    );
  }
}
