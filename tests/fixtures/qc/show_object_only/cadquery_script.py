import cadquery as cq

result = cq.Workplane("XY").box(10, 10, 10)
show_object(result)
