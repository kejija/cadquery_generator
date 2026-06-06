import cadquery as cq

left = cq.Workplane("XY").box(10, 10, 10).translate((-5, 0, 0))
right = cq.Workplane("XY").box(10, 10, 10).translate((5, 0, 0))
result = cq.Assembly()
result.add(left, name="left", color=cq.Color(0.8, 0.8, 0.8))
result.add(right, name="right", color=cq.Color(0.5, 0.5, 0.5))
