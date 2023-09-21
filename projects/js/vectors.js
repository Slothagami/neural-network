var line_param
function init_line(canv) {
    canv.coord_width = 10
    canv.add_draggable(-2, 1, "c", "blue")
}

function line(canv) {
    canv.draw.axes()
    
    let v = new Point(1, 1)
    canv.draw.vector(Origin, canv.drag.c, "blue", "c")
    
    let scale = canv.controls.scale
    canv.draw.vector(canv.drag.c, canv.drag.c.add(v.scale(scale)), "orange", "vt", -30)
    canv.draw.text(3, 2, `t = ${round(scale)}`, "dorange")
}


function colinear(canv) {
    canv.draw.axes()

    let line = t => {
        let base = new Point(1, .5)
        let dir  = new Point(.4, 1)
        return base.add(dir.scale(t))
    }

    let orange = get_color("orange") + "80"
    canv.draw.line(line(-10), line(10), orange, 2)

    for(let i = -3; i < 2; i++) {
        canv.draw.vector(Origin, line(i * .5), "blue")
    }
}

function init_linear_dependance_2d(canv) {
    canv.coord_width = 10
    canv.add_draggable(-2, 3, "target", "white")
}
function linear_dependance_2d(canv) {
    canv.draw.axes()

    let v1 = new Point(2, 1)
    let v2 = new Point(-2, 1/2)

    // convert the basis (matrix multiply)
    let a = new Point(1/6, -1/3).scale(canv.drag.target.x)
    let b = new Point(2/3,  2/3).scale(canv.drag.target.y)
    let new_cord = a.add(b)

    // scale the vectors to sum to the target
    v1 = v1.scale(new_cord.x)
    v2 = v2.scale(new_cord.y)

    canv.draw.point(canv.drag.target, "white")
    canv.draw.vector(Origin, v1, "blue")
    canv.draw.vector(v1, v1.add(v2), "orange")
}


var start = new Point(-6/2, -3/2)
function init_proj_diagram(canv) {
    canv.coord_width = 5

    canv.add_draggable(2, 3, "a", "orange", start)
    canv.add_draggable(6, 1, "b", "blue", start)
}
function proj_diagram(canv) {
    let a = canv.drag.a
    let b = canv.drag.b
    let resolute = b.unit().scale(a.dot(b.unit()))

    canv.draw.angle_between(a, resolute, start, 50, "gray-0")
    canv.draw.vector(start, start.add(a), "orange", "a")
    canv.draw.vector(start, start.add(b), "blue",   "b")
    canv.draw.dottedline(start.add(a), start.add(resolute), "gray-1")
    canv.draw.vector(start, start.add(resolute), "green", "c")
}


function init_dot_prod(canv) {
    canv.coord_width = 5
    canv.add_draggable( 3/2, 3/2, "a", "blue")
    canv.add_draggable(-1, 1, "b", "orange")
}
function dot_prod(canv) {
    canv.draw.angle_between(canv.drag.a, canv.drag.b, Origin, 50, "white")
    canv.draw.vector(Origin, canv.drag.a, "blue", "a")
    canv.draw.vector(Origin, canv.drag.b, "orange", "b")
    canv.draw.text(0, -1, `a . b = ${round(canv.drag.a.dot(canv.drag.b))}`, "gray-1")
}



const graph = t => {
    return new Point(t, -2*(t**2 - t**3))
        .add(new Point(-2/3, 0))
}
function init_derivative(canv) {
    canv.coord_width = 1
}
function derivative(canv) {
    canv.draw.parametric_curve(graph, -4, 4, "gray-0", 4)
    
    let start_val = .7
    let start = graph(start_val)
    let end   = graph(start_val + canv.controls.dt)
    let delta = end.sub(start)
    let dx    = start.add(new Point(delta.x, 0))
    let dr_dt = delta.scale(1/canv.controls.dt * .6)

    canv.draw.text(0, .2, `dt = ${round(canv.controls.dt, 3)}`, "gray-1")

    canv.draw.point(start)
    canv.draw.vector(start, dx, "blue", "dx", -50)
    canv.draw.vector(dx, dx.add(new Point(0, delta.y)), "orange", "dy")

    canv.draw.vector(start, start.add(dr_dt), "purple", "dr/dt", -50)
    canv.draw.vector(start, end, "green", "dr", -50)
}


function init_arclength(canv) {
    canv.coord_width = 1
    canv.controls.dt.anim_time = 1
}
function arclength(canv) { 
    canv.draw.parametric_curve(graph, -4, 4, "gray-2", 4)
    canv.draw.text(0, .2, `dt = ${round(canv.controls.dt, 3)}`, "gray-1")
    
    for(let t = -4; t < 4; t += canv.controls.dt) {
        let start = graph(t)
        let end   = graph(t + canv.controls.dt)
        canv.draw.vector(start, end, "orange")
    }
}



function init_area(canv) {
    canv.coord_width = 1
    canv.controls.dt.anim_time = .8
}
function area(canv) {
    let orig = new Point(-1/2, -1/4)
    const loop = t => {
        return new Point(
            Math.cos(t),
            .5 * Math.sin(2 * t)
        ).add(orig)
    }

    canv.draw.axes(orig)
    canv.draw.text(-1/3, .15, `dt = ${round(canv.controls.dt, 3)}`, "gray-1")
    canv.draw.parametric_curve(loop, 0, Math.PI*2, "gray-1")

    let px = loop(0).x
    for(let t = 0; t < Math.PI/2; t += canv.controls.dt) {
        // draw rectangle
        let point = loop(t)
        let dx = px - point.x
        canv.draw.fillrect(point.x, point.y, point.x + dx, orig.y, get_color("blue") + "80")
        px = point.x
    }
    
    px = loop(0).x
    for(let t = 0; t < Math.PI/2; t += canv.controls.dt) {
        let point = loop(t)
        let dx = px - point.x
        px = point.x

        // draw measurements on hover
        let mouse = canv.mouse()
        let centered = point.sub(orig)
        if(point.x < mouse.x && mouse.x < point.x + dx) {
            if(point.y > mouse.y && mouse.y > orig.y) {
                canv.draw.fillrect(point.x, point.y, point.x + dx, orig.y, get_color("white") + "10")
                
                canv.draw.line(point, point.sub(centered.y_comp()), "orange", 4, "y(t)", -50)
                canv.draw.line(
                    point.x_comp().add(orig.y_comp()), 
                    point.x_comp().add(new Point(dx,0)).add(orig.y_comp()), 
                    "blue", 4, "dx/dt", -50
                )
            }
        }
    }
}
