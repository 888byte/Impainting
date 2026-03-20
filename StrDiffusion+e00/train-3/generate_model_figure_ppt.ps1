Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Inch([double]$v) {
    return [double]($v * 72.0)
}

function RgbInt([string]$hex) {
    $h = $hex.TrimStart('#')
    $r = [Convert]::ToInt32($h.Substring(0, 2), 16)
    $g = [Convert]::ToInt32($h.Substring(2, 2), 16)
    $b = [Convert]::ToInt32($h.Substring(4, 2), 16)
    return ($r + 256 * $g + 65536 * $b)
}

$COLORS = @{
    ProposedFill  = (RgbInt '#FFF1E3')
    ProposedLine  = (RgbInt '#E4872E')
    BackboneFill  = (RgbInt '#EEF2F7')
    BackboneLine  = (RgbInt '#6F8197')
    CondFill      = (RgbInt '#EDF8EF')
    CondLine      = (RgbInt '#63A66F')
    OptionalFill  = (RgbInt '#F4F5F7')
    OptionalLine  = (RgbInt '#A5ABB5')
    PlaceholderFill = (RgbInt '#F8F8F8')
    PlaceholderLine = (RgbInt '#BFC5CC')
    GoldFill      = (RgbInt '#F7E6B8')
    GoldLine      = (RgbInt '#D9A441')
    Text          = (RgbInt '#26323E')
    LightText     = (RgbInt '#55606B')
    White         = (RgbInt '#FFFFFF')
}

$PP_LAYOUT_BLANK = 12
$MSO_SHAPE_RECT = 1
$MSO_SHAPE_ROUNDED_RECT = 5
$MSO_CONNECTOR_STRAIGHT = 1
$MSO_TRUE = -1
$MSO_FALSE = 0
$DASH_STYLE_DASH = 4
$PP_ALIGN_LEFT = 1
$PP_ALIGN_CENTER = 2
$PP_ALIGN_RIGHT = 3

function Apply-TextStyle {
    param(
        $shape,
        [double]$fontSize = 12,
        [bool]$bold = $false,
        [int]$color = 0,
        [int]$align = 2,
        [string]$fontName = 'Aptos'
    )
    $shape.TextFrame.TextRange.Font.Name = $fontName
    $shape.TextFrame.TextRange.Font.Size = $fontSize
    $shape.TextFrame.TextRange.Font.Bold = $(if ($bold) { $MSO_TRUE } else { $MSO_FALSE })
    $shape.TextFrame.TextRange.Font.Color.RGB = $color
    $shape.TextFrame.TextRange.ParagraphFormat.Alignment = $align
    $shape.TextFrame.MarginLeft = 4
    $shape.TextFrame.MarginRight = 4
    $shape.TextFrame.MarginTop = 3
    $shape.TextFrame.MarginBottom = 3
    $shape.TextFrame.WordWrap = $MSO_TRUE
}

function Add-Box {
    param(
        $slide,
        [double]$x,
        [double]$y,
        [double]$w,
        [double]$h,
        [string]$text,
        [int]$fillColor,
        [int]$lineColor,
        [double]$fontSize = 12,
        [bool]$bold = $true,
        [int]$align = 2,
        [int]$shapeType = 5,
        [double]$lineWeight = 1.5,
        [bool]$dashed = $false,
        [int]$textColor = 0
    )
    $s = $slide.Shapes.AddShape($shapeType, (Inch $x), (Inch $y), (Inch $w), (Inch $h))
    $s.Fill.ForeColor.RGB = $fillColor
    $s.Line.ForeColor.RGB = $lineColor
    $s.Line.Weight = $lineWeight
    if ($dashed) { $s.Line.DashStyle = $DASH_STYLE_DASH }
    $s.TextFrame.TextRange.Text = $text
    Apply-TextStyle -shape $s -fontSize $fontSize -bold $bold -color $(if ($textColor -eq 0) { $COLORS.Text } else { $textColor }) -align $align
    return $s
}

function Add-Text {
    param(
        $slide,
        [double]$x,
        [double]$y,
        [double]$w,
        [double]$h,
        [string]$text,
        [double]$fontSize = 12,
        [bool]$bold = $false,
        [int]$color = 0,
        [int]$align = 1
    )
    $s = $slide.Shapes.AddTextbox(1, (Inch $x), (Inch $y), (Inch $w), (Inch $h))
    $s.Line.Visible = $MSO_FALSE
    $s.Fill.Visible = $MSO_FALSE
    $s.TextFrame.TextRange.Text = $text
    Apply-TextStyle -shape $s -fontSize $fontSize -bold $bold -color $(if ($color -eq 0) { $COLORS.Text } else { $color }) -align $align
    return $s
}

function Add-Arrow {
    param(
        $slide,
        [double]$x1,
        [double]$y1,
        [double]$x2,
        [double]$y2,
        [int]$color,
        [double]$weight = 1.8,
        [bool]$dashed = $false
    )
    $c = $slide.Shapes.AddConnector($MSO_CONNECTOR_STRAIGHT, (Inch $x1), (Inch $y1), (Inch $x2), (Inch $y2))
    $c.Line.ForeColor.RGB = $color
    $c.Line.Weight = $weight
    $c.Line.EndArrowheadStyle = 3
    if ($dashed) { $c.Line.DashStyle = $DASH_STYLE_DASH }
    return $c
}

function Add-Title {
    param($slide, [string]$title, [string]$figLabel = '')
    Add-Text -slide $slide -x 0.55 -y 0.18 -w 9.40 -h 0.35 -text $title -fontSize 24 -bold $true -color $COLORS.Text -align $PP_ALIGN_LEFT | Out-Null
    if ($figLabel) {
        Add-Text -slide $slide -x 11.35 -y 0.20 -w 1.35 -h 0.28 -text $figLabel -fontSize 10.5 -bold $true -color $COLORS.LightText -align $PP_ALIGN_RIGHT | Out-Null
    }
    $line = $slide.Shapes.AddShape($MSO_SHAPE_RECT, (Inch 0.55), (Inch 0.56), (Inch 12.15), (Inch 0.03))
    $line.Fill.ForeColor.RGB = $COLORS.BackboneLine
    $line.Line.Visible = $MSO_FALSE
}

function Add-Panel {
    param(
        $slide,
        [double]$x,
        [double]$y,
        [double]$w,
        [double]$h,
        [string]$title,
        [int]$fillColor,
        [int]$lineColor
    )
    $panel = Add-Box -slide $slide -x $x -y $y -w $w -h $h -text '' -fillColor $fillColor -lineColor $lineColor -shapeType $MSO_SHAPE_ROUNDED_RECT -lineWeight 1.4
    Add-Text -slide $slide -x ($x + 0.18) -y ($y + 0.08) -w ($w - 0.30) -h 0.24 -text $title -fontSize 14 -bold $true -color $COLORS.Text -align $PP_ALIGN_LEFT | Out-Null
    return $panel
}

function Add-BulletList {
    param(
        $slide,
        [double]$x,
        [double]$y,
        [double]$w,
        [double]$h,
        [string[]]$items,
        [double]$fontSize = 11,
        [int]$color = 0
    )
    $text = ($items | ForEach-Object { "- $_" }) -join "`r`n"
    Add-Text -slide $slide -x $x -y $y -w $w -h $h -text $text -fontSize $fontSize -bold $false -color $(if ($color -eq 0) { $COLORS.Text } else { $color }) -align $PP_ALIGN_LEFT | Out-Null
}

function Add-StageChip {
    param($slide, [double]$x, [double]$y, [double]$w, [double]$h, [string]$text)
    Add-Box -slide $slide -x $x -y $y -w $w -h $h -text $text -fillColor $COLORS.ProposedFill -lineColor $COLORS.ProposedLine -fontSize 11 -bold $true -shapeType $MSO_SHAPE_ROUNDED_RECT | Out-Null
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$outPath = Join-Path $root 'Proposed_Two_Stage_Model_Figure_Atlas.pptx'

$ppt = $null
$pres = $null
try {
    $ppt = New-Object -ComObject PowerPoint.Application
    $ppt.Visible = $MSO_TRUE
    $pres = $ppt.Presentations.Add()
    $pres.PageSetup.SlideWidth = (Inch 13.333)
    $pres.PageSetup.SlideHeight = (Inch 7.5)

    # Slide 1: cover
    $slide = $pres.Slides.Add(1, $PP_LAYOUT_BLANK)
    $bg = $slide.Shapes.AddShape($MSO_SHAPE_RECT, 0, 0, (Inch 13.333), (Inch 7.5))
    $bg.Fill.ForeColor.RGB = $COLORS.White
    $bg.Line.Visible = $MSO_FALSE
    $banner = $slide.Shapes.AddShape($MSO_SHAPE_ROUNDED_RECT, (Inch 0.7), (Inch 1.1), (Inch 11.9), (Inch 4.8))
    $banner.Fill.ForeColor.RGB = $COLORS.BackboneFill
    $banner.Line.ForeColor.RGB = $COLORS.BackboneLine
    $banner.Line.Weight = 1.6
    Add-Text -slide $slide -x 1.05 -y 1.60 -w 10.90 -h 0.65 -text 'Two-Stage Model Figure Atlas' -fontSize 28 -bold $true -color $COLORS.Text -align $PP_ALIGN_LEFT | Out-Null
    Add-Text -slide $slide -x 1.05 -y 2.28 -w 10.90 -h 0.55 -text 'From SSD-TS Pigment Restoration to Prior-Guided StrDiffusion Inpainting' -fontSize 18 -bold $false -color $COLORS.LightText -align $PP_ALIGN_LEFT | Out-Null
    Add-StageChip -slide $slide -x 1.10 -y 3.05 -w 3.70 -h 0.55 -text 'Stage I: Pigment Restoration Prior Stage'
    Add-StageChip -slide $slide -x 4.98 -y 3.05 -w 4.15 -h 0.55 -text 'Stage II: Prior-Guided Mural Inpainting Stage'
    Add-StageChip -slide $slide -x 9.36 -y 3.05 -w 2.05 -h 0.55 -text 'PPT Draft'
    Add-BulletList -slide $slide -x 1.12 -y 3.95 -w 9.40 -h 1.45 -items @(
        'structure-first figure drafting',
        'paper-ready English labels',
        'top-conference layout placeholders',
        'core active-path modules only'
    ) -fontSize 13
    Add-Text -slide $slide -x 1.05 -y 6.25 -w 6.50 -h 0.28 -text 'Generated from real code paths and atlas specifications' -fontSize 11 -bold $false -color $COLORS.LightText -align $PP_ALIGN_LEFT | Out-Null

    # Slide 2: overall two-stage
    $slide = $pres.Slides.Add(2, $PP_LAYOUT_BLANK)
    Add-Title -slide $slide -title 'Overall Pipeline of the Proposed Two-Stage Restoration Framework' -figLabel 'Fig.1'
    Add-Panel -slide $slide -x 0.55 -y 0.92 -w 5.25 -h 4.95 -title '(A) Stage I: Pigment Restoration Prior Stage' -fillColor $COLORS.BackboneFill -lineColor $COLORS.BackboneLine | Out-Null
    Add-Panel -slide $slide -x 7.12 -y 0.92 -w 5.65 -h 4.95 -title '(B) Stage II: Prior-Guided Mural Inpainting Stage' -fillColor $COLORS.BackboneFill -lineColor $COLORS.BackboneLine | Out-Null
    Add-Box -slide $slide -x 0.90 -y 2.15 -w 0.95 -h 0.80 -text 'Pigment Sample Placeholder' -fillColor $COLORS.PlaceholderFill -lineColor $COLORS.PlaceholderLine -fontSize 9.5 -bold $false -align $PP_ALIGN_CENTER | Out-Null
    Add-Box -slide $slide -x 2.10 -y 2.10 -w 1.15 -h 0.90 -text 'RGB-Only Spectral Bridge' -fillColor $COLORS.ProposedFill -lineColor $COLORS.ProposedLine -fontSize 11.2 | Out-Null
    Add-Box -slide $slide -x 3.60 -y 2.10 -w 1.25 -h 0.90 -text 'Conditional Pigment Diffusion Denoiser' -fillColor $COLORS.BackboneFill -lineColor $COLORS.BackboneLine -fontSize 10.3 | Out-Null
    Add-Box -slide $slide -x 4.20 -y 1.25 -w 1.20 -h 0.62 -text 'Multimodal Teacher Conditioning' -fillColor $COLORS.OptionalFill -lineColor $COLORS.OptionalLine -fontSize 9.3 -bold $false -dashed $true | Out-Null
    Add-Box -slide $slide -x 4.05 -y 3.35 -w 1.20 -h 0.80 -text '3D Pigment LUT Builder' -fillColor $COLORS.ProposedFill -lineColor $COLORS.ProposedLine -fontSize 10.3 | Out-Null
    Add-Arrow -slide $slide -x1 1.85 -y1 2.55 -x2 2.10 -y2 2.55 -color $COLORS.BackboneLine | Out-Null
    Add-Arrow -slide $slide -x1 3.25 -y1 2.55 -x2 3.60 -y2 2.55 -color $COLORS.BackboneLine | Out-Null
    Add-Arrow -slide $slide -x1 4.25 -y1 1.87 -x2 4.25 -y2 2.12 -color $COLORS.OptionalLine -dashed $true | Out-Null
    Add-Arrow -slide $slide -x1 4.25 -y1 3.00 -x2 4.25 -y2 3.35 -color $COLORS.BackboneLine | Out-Null
    $gold = Add-Box -slide $slide -x 5.95 -y 2.55 -w 0.95 -h 1.10 -text 'Pigment LUT / LUT-based Color Knowledge' -fillColor $COLORS.GoldFill -lineColor $COLORS.GoldLine -fontSize 10 -bold $true -shapeType $MSO_SHAPE_ROUNDED_RECT
    $gold.Rotation = 90
    Add-Box -slide $slide -x 7.45 -y 2.10 -w 0.95 -h 0.82 -text 'Mural Input Placeholder' -fillColor $COLORS.PlaceholderFill -lineColor $COLORS.PlaceholderLine -fontSize 9.5 -bold $false | Out-Null
    Add-Box -slide $slide -x 8.70 -y 1.70 -w 1.10 -h 0.72 -text 'Hole Mask' -fillColor $COLORS.CondFill -lineColor $COLORS.CondLine -fontSize 10.5 | Out-Null
    Add-Box -slide $slide -x 8.80 -y 2.75 -w 1.25 -h 0.82 -text 'LUT Prior Composer' -fillColor $COLORS.ProposedFill -lineColor $COLORS.ProposedLine -fontSize 11 | Out-Null
    Add-Box -slide $slide -x 10.25 -y 1.70 -w 1.05 -h 0.72 -text 'Mu Cleaner' -fillColor $COLORS.ProposedFill -lineColor $COLORS.ProposedLine -fontSize 10.8 | Out-Null
    Add-Box -slide $slide -x 10.20 -y 2.75 -w 1.55 -h 0.92 -text 'Prior-Guided Texture U-Net' -fillColor $COLORS.BackboneFill -lineColor $COLORS.BackboneLine -fontSize 10.5 | Out-Null
    Add-Box -slide $slide -x 12.00 -y 2.50 -w 0.55 -h 1.35 -text 'Official-Compatible Enhanced Reverse SDE' -fillColor $COLORS.BackboneFill -lineColor $COLORS.BackboneLine -fontSize 8.6 | Out-Null
    Add-Box -slide $slide -x 11.80 -y 4.25 -w 0.75 -h 0.78 -text 'Final Restoration Placeholder' -fillColor $COLORS.PlaceholderFill -lineColor $COLORS.PlaceholderLine -fontSize 8.8 -bold $false | Out-Null
    Add-Arrow -slide $slide -x1 8.40 -y1 2.50 -x2 8.80 -y2 3.05 -color $COLORS.CondLine | Out-Null
    Add-Arrow -slide $slide -x1 9.25 -y1 2.42 -x2 9.25 -y2 2.75 -color $COLORS.CondLine | Out-Null
    Add-Arrow -slide $slide -x1 10.05 -y1 3.15 -x2 10.20 -y2 3.15 -color $COLORS.BackboneLine | Out-Null
    Add-Arrow -slide $slide -x1 11.75 -y1 3.10 -x2 12.00 -y2 3.10 -color $COLORS.BackboneLine | Out-Null
    Add-Arrow -slide $slide -x1 12.25 -y1 3.85 -x2 12.18 -y2 4.25 -color $COLORS.BackboneLine | Out-Null
    Add-Box -slide $slide -x 0.70 -y 6.05 -w 5.00 -h 0.70 -text 'Stage I detail: RGB-only bridge and LUT construction' -fillColor $COLORS.OptionalFill -lineColor $COLORS.OptionalLine -fontSize 10.5 -bold $false -align $PP_ALIGN_LEFT | Out-Null
    Add-Box -slide $slide -x 7.25 -y 6.05 -w 5.20 -h 0.70 -text 'Stage II detail: prior injection, MGLC, and enhanced reverse SDE' -fillColor $COLORS.OptionalFill -lineColor $COLORS.OptionalLine -fontSize 10.5 -bold $false -align $PP_ALIGN_LEFT | Out-Null

    # Slide 3: baseline reference map
    $slide = $pres.Slides.Add(3, $PP_LAYOUT_BLANK)
    Add-Title -slide $slide -title 'Dual-Baseline Reference Map: From SSD-TS and StrDiffusion to Our Full System' -figLabel 'Fig.2'
    Add-Panel -slide $slide -x 0.60 -y 1.00 -w 3.55 -h 4.85 -title '(A) SSD-TS Baseline Skeleton' -fillColor $COLORS.BackboneFill -lineColor $COLORS.BackboneLine | Out-Null
    Add-Panel -slide $slide -x 4.55 -y 1.00 -w 4.15 -h 4.85 -title '(B) Our Added Modules' -fillColor $COLORS.White -lineColor $COLORS.ProposedLine | Out-Null
    Add-Panel -slide $slide -x 9.10 -y 1.00 -w 3.60 -h 4.85 -title '(C) StrDiffusion Baseline Skeleton' -fillColor $COLORS.BackboneFill -lineColor $COLORS.BackboneLine | Out-Null
    Add-BulletList -slide $slide -x 0.90 -y 1.70 -w 2.90 -h 3.40 -items @(
        'generic conditional diffusion backbone',
        'short-sequence denoiser',
        'basic condition injection',
        'no prototype bank',
        'no LUT builder'
    )
    Add-StageChip -slide $slide -x 4.95 -y 1.62 -w 3.20 -h 0.52 -text 'Stage I Additions'
    Add-BulletList -slide $slide -x 4.95 -y 2.20 -w 3.25 -h 1.55 -items @(
        'Color Evidence Encoder',
        'Pseudo-Spectrum Predictor',
        'Prototype Posterior Estimator',
        'Spectral Prototype Bank',
        'Retrieval Branch and Confidence Gate',
        '3D Pigment LUT Builder'
    ) -fontSize 10.8
    Add-StageChip -slide $slide -x 4.95 -y 4.00 -w 3.20 -h 0.52 -text 'Stage II Additions'
    Add-BulletList -slide $slide -x 4.95 -y 4.58 -w 3.25 -h 1.10 -items @(
        'LUT Prior Composer',
        'Pixel Condition Encoder',
        'MGLC',
        'Mu Cleaner',
        'Enhanced reverse SDE'
    ) -fontSize 10.8
    Add-BulletList -slide $slide -x 9.42 -y 1.70 -w 2.95 -h 3.40 -items @(
        'Conditional UNet',
        'structure branch',
        'discriminator branch',
        'reverse SDE',
        'official inference skeleton'
    )
    Add-Box -slide $slide -x 0.70 -y 6.00 -w 12.10 -h 0.62 -text 'Baseline panels remain blue-gray; all proposed modules are highlighted in orange as additive upgrades rather than backbone replacement.' -fillColor $COLORS.OptionalFill -lineColor $COLORS.OptionalLine -fontSize 10.4 -bold $false -align $PP_ALIGN_LEFT | Out-Null

    # Slide 4: RGB-only bridge
    $slide = $pres.Slides.Add(4, $PP_LAYOUT_BLANK)
    Add-Title -slide $slide -title 'RGB-Only Spectral Bridge for Pseudo Spectral Conditioning' -figLabel 'Fig.6'
    Add-Box -slide $slide -x 0.62 -y 2.42 -w 1.15 -h 1.05 -text 'Input RGB Placeholder' -fillColor $COLORS.PlaceholderFill -lineColor $COLORS.PlaceholderLine -fontSize 9.4 -bold $false | Out-Null
    Add-Box -slide $slide -x 1.95 -y 2.35 -w 1.55 -h 1.12 -text 'Color Evidence Encoder' -fillColor $COLORS.ProposedFill -lineColor $COLORS.ProposedLine -fontSize 10.9 | Out-Null
    Add-Box -slide $slide -x 3.82 -y 1.55 -w 1.80 -h 0.95 -text 'Pseudo-Spectrum Predictor' -fillColor $COLORS.ProposedFill -lineColor $COLORS.ProposedLine -fontSize 10.6 | Out-Null
    Add-Box -slide $slide -x 3.82 -y 3.35 -w 1.80 -h 0.95 -text 'Prototype Posterior Estimator' -fillColor $COLORS.ProposedFill -lineColor $COLORS.ProposedLine -fontSize 10.4 | Out-Null
    Add-Box -slide $slide -x 6.00 -y 0.95 -w 1.65 -h 1.25 -text 'Prototype Bank Placeholder' -fillColor $COLORS.PlaceholderFill -lineColor $COLORS.PlaceholderLine -fontSize 9.5 -bold $false | Out-Null
    Add-Text -slide $slide -x 6.10 -y 0.70 -w 1.60 -h 0.22 -text 'Spectral Prototype Bank' -fontSize 10.5 -bold $true -color $COLORS.Text -align $PP_ALIGN_CENTER | Out-Null
    Add-Box -slide $slide -x 6.05 -y 2.25 -w 1.55 -h 0.95 -text 'Retrieval Branch' -fillColor $COLORS.ProposedFill -lineColor $COLORS.ProposedLine -fontSize 10.7 | Out-Null
    Add-Box -slide $slide -x 8.05 -y 2.25 -w 1.55 -h 1.15 -text 'Posterior-Retrieval Confidence Gate' -fillColor $COLORS.ProposedFill -lineColor $COLORS.ProposedLine -fontSize 9.8 | Out-Null
    Add-Box -slide $slide -x 10.00 -y 2.32 -w 1.45 -h 1.00 -text 'Pseudo Spectral Condition' -fillColor $COLORS.CondFill -lineColor $COLORS.CondLine -fontSize 10.4 | Out-Null
    Add-Box -slide $slide -x 11.55 -y 2.10 -w 1.15 -h 1.45 -text 'Conditional Pigment Diffusion Denoiser' -fillColor $COLORS.BackboneFill -lineColor $COLORS.BackboneLine -fontSize 9.3 | Out-Null
    Add-Arrow -slide $slide -x1 1.77 -y1 2.95 -x2 1.95 -y2 2.95 -color $COLORS.BackboneLine | Out-Null
    Add-Arrow -slide $slide -x1 3.50 -y1 2.85 -x2 3.82 -y2 2.00 -color $COLORS.ProposedLine | Out-Null
    Add-Arrow -slide $slide -x1 3.50 -y1 2.98 -x2 3.82 -y2 3.80 -color $COLORS.ProposedLine | Out-Null
    Add-Arrow -slide $slide -x1 6.85 -y1 2.20 -x2 6.85 -y2 2.25 -color $COLORS.ProposedLine | Out-Null
    Add-Arrow -slide $slide -x1 5.62 -y1 2.02 -x2 8.05 -y2 2.60 -color $COLORS.ProposedLine | Out-Null
    Add-Arrow -slide $slide -x1 5.62 -y1 3.82 -x2 8.05 -y2 2.95 -color $COLORS.ProposedLine | Out-Null
    Add-Arrow -slide $slide -x1 7.60 -y1 2.72 -x2 8.05 -y2 2.72 -color $COLORS.ProposedLine | Out-Null
    Add-Arrow -slide $slide -x1 9.60 -y1 2.82 -x2 10.00 -y2 2.82 -color $COLORS.CondLine | Out-Null
    Add-Arrow -slide $slide -x1 11.45 -y1 2.82 -x2 11.55 -y2 2.82 -color $COLORS.BackboneLine | Out-Null
    Add-Box -slide $slide -x 0.80 -y 5.45 -w 11.85 -h 0.75 -text 'Training note: true spectral condition is available only during training and is not drawn as deployment-time input.' -fillColor $COLORS.OptionalFill -lineColor $COLORS.OptionalLine -fontSize 10.4 -bold $false -align $PP_ALIGN_LEFT | Out-Null

    # Slide 5: LUT construction
    $slide = $pres.Slides.Add(5, $PP_LAYOUT_BLANK)
    Add-Title -slide $slide -title 'Stage-I Deployment Product: 3D Pigment LUT Construction' -figLabel 'Fig.10'
    Add-Box -slide $slide -x 0.85 -y 2.20 -w 2.05 -h 1.35 -text 'RGB Grid Sampling' -fillColor $COLORS.BackboneFill -lineColor $COLORS.BackboneLine -fontSize 11.8 | Out-Null
    Add-Box -slide $slide -x 3.25 -y 2.20 -w 2.35 -h 1.35 -text 'Batch Single-Color Inference' -fillColor $COLORS.ProposedFill -lineColor $COLORS.ProposedLine -fontSize 11.4 | Out-Null
    Add-Box -slide $slide -x 5.95 -y 2.20 -w 2.45 -h 1.35 -text 'Confidence / Uncertainty Diagnostics' -fillColor $COLORS.CondFill -lineColor $COLORS.CondLine -fontSize 10.8 | Out-Null
    Add-Box -slide $slide -x 8.70 -y 2.20 -w 1.85 -h 1.35 -text 'Optional Stabilization' -fillColor $COLORS.OptionalFill -lineColor $COLORS.OptionalLine -fontSize 11.0 -bold $false | Out-Null
    Add-Box -slide $slide -x 10.85 -y 1.92 -w 1.95 -h 1.90 -text '3D LUT Cube Placeholder' -fillColor $COLORS.PlaceholderFill -lineColor $COLORS.PlaceholderLine -fontSize 9.4 -bold $false | Out-Null
    Add-Text -slide $slide -x 10.93 -y 1.60 -w 1.80 -h 0.24 -text '3D Pigment LUT Builder' -fontSize 11.2 -bold $true -color $COLORS.Text -align $PP_ALIGN_CENTER | Out-Null
    Add-Arrow -slide $slide -x1 2.90 -y1 2.88 -x2 3.25 -y2 2.88 -color $COLORS.BackboneLine | Out-Null
    Add-Arrow -slide $slide -x1 5.60 -y1 2.88 -x2 5.95 -y2 2.88 -color $COLORS.BackboneLine | Out-Null
    Add-Arrow -slide $slide -x1 8.40 -y1 2.88 -x2 8.70 -y2 2.88 -color $COLORS.BackboneLine | Out-Null
    Add-Arrow -slide $slide -x1 10.55 -y1 2.88 -x2 10.85 -y2 2.88 -color $COLORS.GoldLine -weight 2.2 | Out-Null
    Add-Box -slide $slide -x 9.55 -y 4.35 -w 3.10 -h 1.05 -text 'Stored keys: lut_rgb, lut_lab, lut_conf, lut_std, lut_cdiff, lut_cret' -fillColor $COLORS.OptionalFill -lineColor $COLORS.OptionalLine -fontSize 9.8 -bold $false -align $PP_ALIGN_LEFT | Out-Null

    # Slide 6: enhanced stage II
    $slide = $pres.Slides.Add(6, $PP_LAYOUT_BLANK)
    Add-Title -slide $slide -title 'Enhanced Stage-II Framework on Top of StrDiffusion' -figLabel 'Fig.12'
    Add-Panel -slide $slide -x 0.55 -y 0.95 -w 9.05 -h 5.05 -title '(A) Main Stage-II Architecture' -fillColor $COLORS.BackboneFill -lineColor $COLORS.BackboneLine | Out-Null
    Add-Box -slide $slide -x 0.90 -y 2.40 -w 1.10 -h 0.90 -text 'Degraded Mural' -fillColor $COLORS.PlaceholderFill -lineColor $COLORS.PlaceholderLine -fontSize 10.0 -bold $false | Out-Null
    Add-Box -slide $slide -x 1.20 -y 1.35 -w 0.95 -h 0.65 -text 'Hole Mask' -fillColor $COLORS.CondFill -lineColor $COLORS.CondLine -fontSize 10.3 | Out-Null
    Add-Box -slide $slide -x 2.35 -y 2.30 -w 1.35 -h 1.00 -text 'LUT Prior Composer' -fillColor $COLORS.ProposedFill -lineColor $COLORS.ProposedLine -fontSize 10.7 | Out-Null
    Add-Box -slide $slide -x 2.45 -y 3.75 -w 1.15 -h 0.78 -text 'Mu Cleaner' -fillColor $COLORS.ProposedFill -lineColor $COLORS.ProposedLine -fontSize 10.5 | Out-Null
    Add-Box -slide $slide -x 4.10 -y 1.85 -w 1.35 -h 0.85 -text 'Pixel Condition Encoder' -fillColor $COLORS.ProposedFill -lineColor $COLORS.ProposedLine -fontSize 10.2 | Out-Null
    Add-Panel -slide $slide -x 5.70 -y 1.65 -w 2.15 -h 2.80 -title 'Prior-Guided Texture U-Net' -fillColor $COLORS.White -lineColor $COLORS.BackboneLine | Out-Null
    Add-Box -slide $slide -x 5.95 -y 2.45 -w 0.65 -h 0.58 -text 'Enc-1' -fillColor $COLORS.BackboneFill -lineColor $COLORS.BackboneLine -fontSize 8.8 | Out-Null
    Add-Box -slide $slide -x 6.75 -y 2.10 -w 0.70 -h 0.58 -text 'Enc-2' -fillColor $COLORS.BackboneFill -lineColor $COLORS.BackboneLine -fontSize 8.8 | Out-Null
    Add-Box -slide $slide -x 6.75 -y 3.05 -w 0.70 -h 0.58 -text 'MGLC' -fillColor $COLORS.ProposedFill -lineColor $COLORS.ProposedLine -fontSize 9.2 | Out-Null
    Add-Box -slide $slide -x 8.10 -y 2.35 -w 1.20 -h 1.35 -text 'Official-Compatible Enhanced Reverse SDE' -fillColor $COLORS.BackboneFill -lineColor $COLORS.BackboneLine -fontSize 9.0 | Out-Null
    Add-Box -slide $slide -x 7.95 -y 0.98 -w 1.45 -h 0.55 -text 'Optional Structure Guidance' -fillColor $COLORS.OptionalFill -lineColor $COLORS.OptionalLine -fontSize 9.4 -bold $false -dashed $true | Out-Null
    Add-Box -slide $slide -x 8.35 -y 4.80 -w 0.95 -h 0.88 -text 'Restored Mural' -fillColor $COLORS.PlaceholderFill -lineColor $COLORS.PlaceholderLine -fontSize 9.5 -bold $false | Out-Null
    Add-Arrow -slide $slide -x1 2.00 -y1 2.85 -x2 2.35 -y2 2.85 -color $COLORS.BackboneLine | Out-Null
    Add-Arrow -slide $slide -x1 1.68 -y1 2.00 -x2 2.55 -y2 2.30 -color $COLORS.CondLine | Out-Null
    Add-Arrow -slide $slide -x1 3.70 -y1 2.75 -x2 4.10 -y2 2.28 -color $COLORS.ProposedLine | Out-Null
    Add-Arrow -slide $slide -x1 3.60 -y1 4.00 -x2 5.70 -y2 4.00 -color $COLORS.ProposedLine | Out-Null
    Add-Arrow -slide $slide -x1 5.45 -y1 2.28 -x2 5.70 -y2 2.28 -color $COLORS.ProposedLine | Out-Null
    Add-Arrow -slide $slide -x1 7.85 -y1 3.05 -x2 8.10 -y2 3.05 -color $COLORS.BackboneLine | Out-Null
    Add-Arrow -slide $slide -x1 8.70 -y1 1.53 -x2 8.70 -y2 2.35 -color $COLORS.OptionalLine -dashed $true | Out-Null
    Add-Arrow -slide $slide -x1 8.70 -y1 3.70 -x2 8.70 -y2 4.80 -color $COLORS.BackboneLine | Out-Null
    Add-Box -slide $slide -x 9.95 -y 1.02 -w 2.75 -h 1.45 -text 'Callout 1: Pixel Condition Encoder is a side branch aligned with the main U-Net.' -fillColor $COLORS.OptionalFill -lineColor $COLORS.OptionalLine -fontSize 10.1 -bold $false -align $PP_ALIGN_LEFT | Out-Null
    Add-Box -slide $slide -x 9.95 -y 2.78 -w 2.75 -h 1.45 -text 'Callout 2: MGLC is inserted after prior fusion inside bottleneck / decoder.' -fillColor $COLORS.OptionalFill -lineColor $COLORS.OptionalLine -fontSize 10.1 -bold $false -align $PP_ALIGN_LEFT | Out-Null
    Add-Box -slide $slide -x 9.95 -y 4.54 -w 2.75 -h 1.45 -text 'Callout 3: Inference remains compatible by keeping the official reverse_sde(...) entry.' -fillColor $COLORS.OptionalFill -lineColor $COLORS.OptionalLine -fontSize 10.1 -bold $false -align $PP_ALIGN_LEFT | Out-Null

    # Slide 7: LUT prior composer
    $slide = $pres.Slides.Add(7, $PP_LAYOUT_BLANK)
    Add-Title -slide $slide -title 'LUT Prior Composer for Color Prior and Confidence Construction' -figLabel 'Fig.13'
    Add-Box -slide $slide -x 0.75 -y 2.35 -w 1.15 -h 1.00 -text 'Mural Input Placeholder' -fillColor $COLORS.PlaceholderFill -lineColor $COLORS.PlaceholderLine -fontSize 9.2 -bold $false | Out-Null
    Add-Box -slide $slide -x 0.75 -y 1.05 -w 1.15 -h 0.95 -text 'LUT Cube Placeholder' -fillColor $COLORS.PlaceholderFill -lineColor $COLORS.PlaceholderLine -fontSize 9.2 -bold $false | Out-Null
    Add-Box -slide $slide -x 2.20 -y 2.15 -w 1.75 -h 1.20 -text 'Trilinear LUT Mapper' -fillColor $COLORS.ProposedFill -lineColor $COLORS.ProposedLine -fontSize 11.0 | Out-Null
    Add-Box -slide $slide -x 4.35 -y 2.15 -w 1.75 -h 1.20 -text 'Hole-Region Inpainting' -fillColor $COLORS.ProposedFill -lineColor $COLORS.ProposedLine -fontSize 11.0 | Out-Null
    Add-Box -slide $slide -x 6.50 -y 2.15 -w 1.95 -h 1.20 -text 'Spatial Confidence Estimation' -fillColor $COLORS.CondFill -lineColor $COLORS.CondLine -fontSize 10.8 | Out-Null
    Add-Box -slide $slide -x 8.90 -y 2.15 -w 1.60 -h 1.20 -text 'Confidence Fusion' -fillColor $COLORS.ProposedFill -lineColor $COLORS.ProposedLine -fontSize 11.0 | Out-Null
    Add-Box -slide $slide -x 10.95 -y 1.65 -w 1.15 -h 0.95 -text 'Color Prior' -fillColor $COLORS.CondFill -lineColor $COLORS.CondLine -fontSize 10.8 | Out-Null
    Add-Box -slide $slide -x 10.95 -y 3.05 -w 1.15 -h 0.95 -text 'Confidence Heatmap Placeholder' -fillColor $COLORS.PlaceholderFill -lineColor $COLORS.PlaceholderLine -fontSize 9.2 -bold $false | Out-Null
    Add-Arrow -slide $slide -x1 1.90 -y1 2.75 -x2 2.20 -y2 2.75 -color $COLORS.BackboneLine | Out-Null
    Add-Arrow -slide $slide -x1 1.90 -y1 1.50 -x2 2.75 -y2 2.15 -color $COLORS.GoldLine | Out-Null
    Add-Arrow -slide $slide -x1 3.95 -y1 2.75 -x2 4.35 -y2 2.75 -color $COLORS.BackboneLine | Out-Null
    Add-Arrow -slide $slide -x1 6.10 -y1 2.75 -x2 6.50 -y2 2.75 -color $COLORS.BackboneLine | Out-Null
    Add-Arrow -slide $slide -x1 8.45 -y1 2.75 -x2 8.90 -y2 2.75 -color $COLORS.CondLine | Out-Null
    Add-Arrow -slide $slide -x1 10.50 -y1 2.50 -x2 10.95 -y2 2.12 -color $COLORS.CondLine | Out-Null
    Add-Arrow -slide $slide -x1 10.50 -y1 2.95 -x2 10.95 -y2 3.42 -color $COLORS.CondLine | Out-Null

    # Slide 8: pixel condition encoder
    $slide = $pres.Slides.Add(8, $PP_LAYOUT_BLANK)
    Add-Title -slide $slide -title 'Pixel Condition Encoder and Multi-Scale Feature Injection' -figLabel 'Fig.14'
    Add-Box -slide $slide -x 0.65 -y 2.30 -w 1.75 -h 1.15 -text 'Noisy Image + mask_hole + Color Prior + Confidence' -fillColor $COLORS.CondFill -lineColor $COLORS.CondLine -fontSize 9.5 | Out-Null
    Add-Box -slide $slide -x 2.75 -y 1.80 -w 2.00 -h 2.15 -text 'Pixel Condition Encoder' -fillColor $COLORS.ProposedFill -lineColor $COLORS.ProposedLine -fontSize 12.0 | Out-Null
    foreach ($i in 0..3) {
        $yy = 1.20 + (0.85 * $i)
        Add-Box -slide $slide -x 5.10 -y $yy -w 1.20 -h 0.62 -text $(if ($i -lt 3) { 'Zero-Conv' } else { 'Mid Proj.' }) -fillColor $COLORS.ProposedFill -lineColor $COLORS.ProposedLine -fontSize 9.0 | Out-Null
    }
    $levels = @('Encoder Level 1','Encoder Level 2','Encoder Level 3','Bottleneck')
    for ($i = 0; $i -lt $levels.Count; $i++) {
        $yy = 1.10 + (0.90 * $i)
        Add-Box -slide $slide -x 7.10 -y $yy -w 2.05 -h 0.80 -text $levels[$i] -fillColor $COLORS.BackboneFill -lineColor $COLORS.BackboneLine -fontSize 10.2 | Out-Null
        Add-Arrow -slide $slide -x1 6.30 -y1 ($yy + 0.30) -x2 7.10 -y2 ($yy + 0.30) -color $COLORS.ProposedLine | Out-Null
    }
    Add-Arrow -slide $slide -x1 2.40 -y1 2.88 -x2 2.75 -y2 2.88 -color $COLORS.CondLine | Out-Null
    Add-Box -slide $slide -x 9.65 -y 1.60 -w 2.55 -h 2.65 -text "Condition channels: noisy image, mask_hole, color prior, confidence`r`n`r`nSide branch is aligned with the main U-Net.`r`n`r`nZero-conv projections enable stable multi-scale injection." -fillColor $COLORS.OptionalFill -lineColor $COLORS.OptionalLine -fontSize 10.0 -bold $false -align $PP_ALIGN_LEFT | Out-Null

    # Slide 9: MGLC
    $slide = $pres.Slides.Add(9, $PP_LAYOUT_BLANK)
    Add-Title -slide $slide -title 'Mask-Gated Local Context Block' -figLabel 'Fig.16'
    Add-Box -slide $slide -x 0.80 -y 2.55 -w 1.15 -h 0.95 -text 'Feature Map' -fillColor $COLORS.BackboneFill -lineColor $COLORS.BackboneLine -fontSize 10.5 | Out-Null
    Add-Box -slide $slide -x 2.35 -y 1.60 -w 1.65 -h 1.00 -text 'Local Branch' -fillColor $COLORS.ProposedFill -lineColor $COLORS.ProposedLine -fontSize 10.8 | Out-Null
    Add-Box -slide $slide -x 2.35 -y 3.55 -w 1.90 -h 1.00 -text 'Context Branch (sem_lite)' -fillColor $COLORS.ProposedFill -lineColor $COLORS.ProposedLine -fontSize 10.0 | Out-Null
    Add-Box -slide $slide -x 2.10 -y 5.05 -w 1.05 -h 0.70 -text 'mask_hole' -fillColor $COLORS.CondFill -lineColor $COLORS.CondLine -fontSize 9.8 | Out-Null
    Add-Box -slide $slide -x 4.65 -y 4.90 -w 1.45 -h 0.80 -text 'Boundary Band' -fillColor $COLORS.CondFill -lineColor $COLORS.CondLine -fontSize 9.8 | Out-Null
    Add-Box -slide $slide -x 5.05 -y 2.55 -w 1.25 -h 1.05 -text 'Mask Gate' -fillColor $COLORS.ProposedFill -lineColor $COLORS.ProposedLine -fontSize 10.5 | Out-Null
    Add-Box -slide $slide -x 7.10 -y 2.50 -w 1.60 -h 1.10 -text 'Residual Fusion' -fillColor $COLORS.BackboneFill -lineColor $COLORS.BackboneLine -fontSize 10.5 | Out-Null
    Add-Box -slide $slide -x 9.15 -y 2.55 -w 1.25 -h 0.95 -text 'Enhanced Feature' -fillColor $COLORS.BackboneFill -lineColor $COLORS.BackboneLine -fontSize 10.3 | Out-Null
    Add-Arrow -slide $slide -x1 1.95 -y1 3.00 -x2 2.35 -y2 2.10 -color $COLORS.BackboneLine | Out-Null
    Add-Arrow -slide $slide -x1 1.95 -y1 3.00 -x2 2.35 -y2 4.05 -color $COLORS.BackboneLine | Out-Null
    Add-Arrow -slide $slide -x1 3.15 -y1 5.05 -x2 4.65 -y2 5.30 -color $COLORS.CondLine | Out-Null
    Add-Arrow -slide $slide -x1 3.95 -y1 2.10 -x2 5.05 -y2 2.95 -color $COLORS.ProposedLine | Out-Null
    Add-Arrow -slide $slide -x1 4.25 -y1 4.05 -x2 5.05 -y2 3.15 -color $COLORS.ProposedLine | Out-Null
    Add-Arrow -slide $slide -x1 5.35 -y1 4.90 -x2 5.60 -y2 3.60 -color $COLORS.CondLine | Out-Null
    Add-Arrow -slide $slide -x1 6.30 -y1 3.05 -x2 7.10 -y2 3.05 -color $COLORS.ProposedLine | Out-Null
    Add-Arrow -slide $slide -x1 8.70 -y1 3.05 -x2 9.15 -y2 3.05 -color $COLORS.BackboneLine | Out-Null
    Add-Box -slide $slide -x 10.75 -y 1.65 -w 1.80 -h 2.80 -text "branch_mode: local / context / both`r`n`r`nsupports sem_lite backend`r`n`r`nused at bottleneck and decoder`r`n`r`nmask gate uses hole region and boundary cues" -fillColor $COLORS.OptionalFill -lineColor $COLORS.OptionalLine -fontSize 9.8 -bold $false -align $PP_ALIGN_LEFT | Out-Null

    # Slide 10: Mu Cleaner
    $slide = $pres.Slides.Add(10, $PP_LAYOUT_BLANK)
    Add-Title -slide $slide -title 'Mu Cleaner Before SDE' -figLabel 'Fig.17'
    Add-Box -slide $slide -x 1.05 -y 2.45 -w 1.60 -h 1.10 -text 'Degraded RGB + mask_known + confidence' -fillColor $COLORS.CondFill -lineColor $COLORS.CondLine -fontSize 9.7 | Out-Null
    Add-Box -slide $slide -x 3.15 -y 2.45 -w 1.90 -h 1.10 -text 'Blind-Spot Corruption' -fillColor $COLORS.OptionalFill -lineColor $COLORS.OptionalLine -fontSize 10.8 -bold $false | Out-Null
    Add-Box -slide $slide -x 5.65 -y 2.30 -w 2.10 -h 1.40 -text 'Mu Cleaner' -fillColor $COLORS.ProposedFill -lineColor $COLORS.ProposedLine -fontSize 12.0 | Out-Null
    Add-Box -slide $slide -x 8.35 -y 2.45 -w 1.35 -h 1.10 -text 'mu_clean' -fillColor $COLORS.CondFill -lineColor $COLORS.CondLine -fontSize 11.0 | Out-Null
    Add-Arrow -slide $slide -x1 2.65 -y1 3.00 -x2 3.15 -y2 3.00 -color $COLORS.CondLine | Out-Null
    Add-Arrow -slide $slide -x1 5.05 -y1 3.00 -x2 5.65 -y2 3.00 -color $COLORS.BackboneLine | Out-Null
    Add-Arrow -slide $slide -x1 7.75 -y1 3.00 -x2 8.35 -y2 3.00 -color $COLORS.ProposedLine | Out-Null
    Add-Box -slide $slide -x 10.10 -y 1.95 -w 2.10 -h 2.10 -text "Inputs: degraded RGB, mask_known, confidence`r`n`r`nSelf-supervised blind-spot training`r`n`r`nOnly known region is preserved`r`n`r`nApplied before state generation / reverse SDE" -fillColor $COLORS.OptionalFill -lineColor $COLORS.OptionalLine -fontSize 9.8 -bold $false -align $PP_ALIGN_LEFT | Out-Null

    # Slide 11: Enhanced reverse SDE
    $slide = $pres.Slides.Add(11, $PP_LAYOUT_BLANK)
    Add-Title -slide $slide -title 'Official-Compatible Enhanced Reverse SDE' -figLabel 'Fig.18'
    Add-Box -slide $slide -x 0.75 -y 2.25 -w 2.05 -h 1.20 -text 'Official reverse_sde(...) Entry' -fillColor $COLORS.BackboneFill -lineColor $COLORS.BackboneLine -fontSize 11.0 | Out-Null
    Add-Box -slide $slide -x 3.30 -y 2.10 -w 2.30 -h 1.50 -text 'Conditioned Score Estimation' -fillColor $COLORS.BackboneFill -lineColor $COLORS.BackboneLine -fontSize 11.2 | Out-Null
    Add-Box -slide $slide -x 6.15 -y 2.30 -w 1.25 -h 1.10 -text 'pred_full' -fillColor $COLORS.BackboneFill -lineColor $COLORS.BackboneLine -fontSize 11.0 | Out-Null
    Add-Box -slide $slide -x 7.95 -y 2.15 -w 1.85 -h 1.40 -text 'partial / full Composition' -fillColor $COLORS.ProposedFill -lineColor $COLORS.ProposedLine -fontSize 10.5 | Out-Null
    Add-Box -slide $slide -x 10.40 -y 2.20 -w 1.55 -h 1.25 -text 'Final Output Placeholder' -fillColor $COLORS.PlaceholderFill -lineColor $COLORS.PlaceholderLine -fontSize 9.3 -bold $false | Out-Null
    Add-Box -slide $slide -x 3.15 -y 0.95 -w 1.20 -h 0.72 -text 'Color Prior' -fillColor $COLORS.CondFill -lineColor $COLORS.CondLine -fontSize 9.5 | Out-Null
    Add-Box -slide $slide -x 4.55 -y 0.95 -w 1.20 -h 0.72 -text 'Confidence' -fillColor $COLORS.CondFill -lineColor $COLORS.CondLine -fontSize 9.5 | Out-Null
    Add-Box -slide $slide -x 5.95 -y 0.95 -w 1.20 -h 0.72 -text 'mask_hole' -fillColor $COLORS.CondFill -lineColor $COLORS.CondLine -fontSize 9.5 | Out-Null
    Add-Box -slide $slide -x 7.55 -y 0.85 -w 1.95 -h 0.88 -text 'Optional Structure Guidance' -fillColor $COLORS.OptionalFill -lineColor $COLORS.OptionalLine -fontSize 9.2 -bold $false -dashed $true | Out-Null
    Add-Box -slide $slide -x 9.80 -y 0.85 -w 2.05 -h 0.88 -text 'Optional Discriminator Guidance' -fillColor $COLORS.OptionalFill -lineColor $COLORS.OptionalLine -fontSize 9.0 -bold $false -dashed $true | Out-Null
    Add-Arrow -slide $slide -x1 2.80 -y1 2.85 -x2 3.30 -y2 2.85 -color $COLORS.BackboneLine | Out-Null
    Add-Arrow -slide $slide -x1 4.00 -y1 1.67 -x2 4.00 -y2 2.10 -color $COLORS.CondLine | Out-Null
    Add-Arrow -slide $slide -x1 5.20 -y1 1.67 -x2 4.80 -y2 2.10 -color $COLORS.CondLine | Out-Null
    Add-Arrow -slide $slide -x1 6.60 -y1 1.67 -x2 5.30 -y2 2.10 -color $COLORS.CondLine | Out-Null
    Add-Arrow -slide $slide -x1 8.50 -y1 1.73 -x2 5.20 -y2 2.05 -color $COLORS.OptionalLine -dashed $true | Out-Null
    Add-Arrow -slide $slide -x1 10.80 -y1 1.73 -x2 5.45 -y2 2.25 -color $COLORS.OptionalLine -dashed $true | Out-Null
    Add-Arrow -slide $slide -x1 5.60 -y1 2.85 -x2 6.15 -y2 2.85 -color $COLORS.BackboneLine | Out-Null
    Add-Arrow -slide $slide -x1 7.40 -y1 2.85 -x2 7.95 -y2 2.85 -color $COLORS.ProposedLine | Out-Null
    Add-Arrow -slide $slide -x1 9.80 -y1 2.85 -x2 10.40 -y2 2.85 -color $COLORS.BackboneLine | Out-Null
    Add-Box -slide $slide -x 0.78 -y 5.30 -w 11.90 -h 0.80 -text 'partial mode: known input + predicted hole    |    full mode: direct full prediction    |    compatibility preserved via official entry' -fillColor $COLORS.OptionalFill -lineColor $COLORS.OptionalLine -fontSize 10.0 -bold $false -align $PP_ALIGN_LEFT | Out-Null

    # Slide 12: training vs inference semantics
    $slide = $pres.Slides.Add(12, $PP_LAYOUT_BLANK)
    Add-Title -slide $slide -title 'Stage-II Training and Inference Semantics' -figLabel 'Fig.19'
    Add-Panel -slide $slide -x 0.75 -y 1.10 -w 5.70 -h 4.95 -title '(A) Training Semantics' -fillColor $COLORS.BackboneFill -lineColor $COLORS.BackboneLine | Out-Null
    Add-Panel -slide $slide -x 6.90 -y 1.10 -w 5.70 -h 4.95 -title '(B) Inference Semantics' -fillColor $COLORS.BackboneFill -lineColor $COLORS.BackboneLine | Out-Null
    Add-BulletList -slide $slide -x 1.05 -y 1.75 -w 5.05 -h 3.90 -items @(
        'dataset mask semantics: 1 = hole',
        'mask_for_sde = 1 - mask',
        'color prior and confidence are prepared before the main generator',
        'mu_clean is computed before random state generation',
        'gt_mode supports full / partial / mixed',
        'mu_denoiser.* is saved with the main checkpoint'
    ) -fontSize 10.8
    Add-BulletList -slide $slide -x 7.20 -y 1.75 -w 5.05 -h 3.90 -items @(
        'mask_known and mask_hole are explicit',
        'prior/confidence can be auto-generated at test time',
        'known-region prior is synced with latest LUT mapping',
        'mu_clean is applied only on known region',
        'partial/full composition is selected at output',
        'intermediate exports support debugging'
    ) -fontSize 10.8
    Add-Box -slide $slide -x 0.90 -y 6.15 -w 11.85 -h 0.48 -text 'Shared semantics: BrushNet and MGLC consume mask_hole; Mu Cleaner and known-region preservation consume mask_known.' -fillColor $COLORS.OptionalFill -lineColor $COLORS.OptionalLine -fontSize 10.0 -bold $false -align $PP_ALIGN_LEFT | Out-Null

    # Slide 13: selection guide
    $slide = $pres.Slides.Add(13, $PP_LAYOUT_BLANK)
    Add-Title -slide $slide -title 'Selection Guide for Paper, Supplementary, and Defense Use' -figLabel 'Fig.22'
    Add-Panel -slide $slide -x 0.75 -y 1.20 -w 3.65 -h 4.80 -title 'Main Paper Figures' -fillColor $COLORS.BackboneFill -lineColor $COLORS.BackboneLine | Out-Null
    Add-Panel -slide $slide -x 4.85 -y 1.20 -w 3.65 -h 4.80 -title 'Supplementary Figures' -fillColor $COLORS.BackboneFill -lineColor $COLORS.BackboneLine | Out-Null
    Add-Panel -slide $slide -x 8.95 -y 1.20 -w 3.65 -h 4.80 -title 'Defense Figures' -fillColor $COLORS.BackboneFill -lineColor $COLORS.BackboneLine | Out-Null
    Add-BulletList -slide $slide -x 1.05 -y 1.90 -w 3.05 -h 3.70 -items @(
        'Fig.1 Overall Two-Stage Framework',
        'Fig.6 RGB-Only Spectral Bridge',
        'Fig.12 Enhanced Stage-II Overview',
        'Fig.14 Pixel Condition Encoder',
        'Fig.18 Enhanced Reverse SDE'
    ) -fontSize 10.8
    Add-BulletList -slide $slide -x 5.15 -y 1.90 -w 3.05 -h 3.70 -items @(
        'Fig.2 Dual-Baseline Map',
        'Fig.10 LUT Construction',
        'Fig.13 LUT Prior Composer',
        'Fig.16 MGLC Anatomy',
        'Fig.19 Train vs Inference'
    ) -fontSize 10.8
    Add-BulletList -slide $slide -x 9.25 -y 1.90 -w 3.05 -h 3.70 -items @(
        'Fig.1 Overall Pipeline',
        'Fig.2 Baseline Comparison',
        'Fig.6 RGB Bridge',
        'Fig.12 Stage-II Overview',
        'Fig.16 MGLC',
        'Fig.18 Reverse SDE'
    ) -fontSize 10.8

    if (Test-Path $outPath) {
        Remove-Item $outPath -Force
    }
    $pres.SaveAs($outPath)
    $pres.Close()
    $ppt.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($pres) | Out-Null
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($ppt) | Out-Null
    Write-Output "PPT_SAVED: $outPath"
}
catch {
    if ($pres -ne $null) {
        try { $pres.Close() } catch {}
    }
    if ($ppt -ne $null) {
        try { $ppt.Quit() } catch {}
    }
    throw
}
