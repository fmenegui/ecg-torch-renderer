"""Build review sheets from actual renderer outputs without modifying their pixels.

Run after capture_scenarios.py and optionally provide its previous-version output.
Only review panels are resized. Native PNGs remain the reference observations.
"""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw


def review(captures, out, before=None):
    captures, out = Path(captures), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = ['crumpled_photo', 'folded_archive', 'coffee_spill', 'water_damage',
             'torn_edges', 'cut_sheet', 'missing_fragments', 'thermal_damage']
    width, height, label = 640, 480, 34
    sheet = Image.new('RGB', (width*2, (height+label)*4), '#efefec')
    draw = ImageDraw.Draw(sheet)
    records = []
    for index, name in enumerate(names):
        path = captures/(name+'.png')
        with Image.open(path) as image:
            x, y = index%2*width, index//2*(height+label)
            draw.text((x+12,y+11), name+' | actual torch output', fill='#222')
            sheet.paste(image.resize((width,height),Image.Resampling.LANCZOS),(x,y+label))
        records.append({'name':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    sheet.save(out/'adversity-gallery.jpg',quality=92,subsampling=0)
    if before:
        names = ['aged_scan','folded_photo','damaged_archive']
        comparison = Image.new('RGB',(width*2,(height+label)*len(names)),'#efefec')
        painter = ImageDraw.Draw(comparison)
        for row,name in enumerate(names):
            for column,directory in enumerate((Path(before),captures)):
                with Image.open(directory/(name+'.png')) as image:
                    x,y=column*width,row*(height+label)
                    caption=('0.7.0' if column==0 else '0.8.0')+' | '+name
                    painter.text((x+12,y+11),caption,fill='#222')
                    comparison.paste(image.resize((width,height),Image.Resampling.LANCZOS),(x,y+label))
        comparison.save(out/'before-after.jpg',quality=92,subsampling=0)
    report={'schema':'ecg-adversity-review-v1','sources':records,
            'panel_resizing':'Lanczos, source PNGs retained unchanged',
            'comparison':'same scenario seeds, revised parameter distributions and resolution',
            'external_real_image_benchmark':False}
    (out/'review.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--before',type=Path)
    args=parser.parse_args()
    review(args.captures,args.out,args.before)
