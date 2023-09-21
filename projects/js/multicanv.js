const time  = () => performance.now()/1000
const round = (n, dp=2) => Math.round(n*10**dp)/(10**dp)
const clamp = (x, min, max) => Math.max(Math.min(max, x), min)
const lerp  = (a, b, perc) => a + (b-a) * perc

class MultiCanv {
    constructor(fps=30, default_width=1, default_ratio=1/3, default_ord_width=4) {
        this.canvases = []
        this.def_width = default_width
        this.def_ratio = default_ratio
        this.def_ord_width = default_ord_width
        this.fps = fps

        this.mouse = new Point(0, 0)
        this.mouse_down = false
        this.scale = 2

        window.addEventListener("resize", ()=>{
            for(let canv of this.canvases) {
                this.resize_canvas(canv)
            }
        })

        const mousestart = () => {this.mouse_down = true}
        const mousestop  = () => {this.mouse_down = false}
        window.addEventListener("mousemove", e => {
            this.mouse.x = e.clientX
            this.mouse.y = e.clientY
        })
        window.addEventListener("touchmove", e => {
            // e.preventDefault()
            this.mouse.x = e.changedTouches[0].clientX
            this.mouse.y = e.changedTouches[0].clientY
        })

        window.addEventListener("mousedown", mousestart)
        window.addEventListener("touchstart", mousestart)
        window.addEventListener("mouseup", mousestop)
        window.addEventListener("touchend", mousestop)
        // window.addEventListener("touchcancel", mousestop)
    }

    mouse_pos(canv) {
        let rect = canv.canvas.getBoundingClientRect()
        let canv_corner = new Point(rect.left, rect.top)
        let canv_pos = this.mouse.sub(canv_corner)

        let canv_center = new Point(
            canv.canvas.width , 
            canv.canvas.height
        ).scale(.5 * 1/this.scale)

        let centered_pos = canv_pos.sub(canv_center)
        let scaled = new Point(
            centered_pos.x / (canv_center.x - (1/this.scale)*canv.draw.margin),
            centered_pos.y / (canv_center.x - (1/this.scale)*canv.draw.margin),
        )

        let pos = new Point(
            scaled.x * canv.coord_width,
            scaled.y * canv.coord_width * -1
        )
        return pos
    }

    start() {
        let frame = () => {
            this.update()
            requestAnimationFrame(frame)
        }
        requestAnimationFrame(frame)
    }

    autosetup() {
        let canvases = document.querySelectorAll("canvas")
        canvases.forEach(canv => {
            if(canv.id) this.add(canv.id)
        })
        this.start()
    }

    add(name, options) {
        let { cordinate_width, ratio, width } = options || {}
        let canv_selector = `#${name}`
        let func = window[name]
        let init = window[`init_${name}`]

        let canvas = document.querySelector(canv_selector)
        let canv = {
            canvas:     canvas,
            c:          canvas.getContext("2d"),
            function:   func,
            width:      width || this.def_width,
            ratio:      ratio || this.def_ratio,
            coord_width: cordinate_width || this.def_ord_width,
            mouse:      () => this.mouse_pos(canv),
            drag: {},
            dragging: false
        }
        canv.add_draggable = (x, y, name, color, offset=Origin) => {new Draggable(x, y, name, canv, color, offset)}
        canv.draw = new CDraw(canv)

        // find controls element if it exists
        let controls = document.querySelector(canv_selector + "-controls")
        if(controls) {
            canv.controls = {}
            canv.buttons  = {}

            // generate coltroler objects
            let inputs = controls.querySelectorAll("input")
            inputs.forEach(el => {
                let ctrl_name = el.id.replace(name + "-", "")
                canv.controls[ctrl_name] = new NumControler(el)
            })

            let buttons = controls.querySelectorAll("button")
            buttons.forEach(el => {
                let ctrl_name = el.id.replace(name + "-", "")
                canv.buttons[ctrl_name] = el
            })
        }

        if(init) init(canv)

        this.resize_canvas(canv)
        this.canvases.push(canv)

        return this
    }

    update() {
        for(let canv of this.canvases) {
            // update only canvases that are visible on screen
            let canv_bbox = canv.canvas.getBoundingClientRect()
            if(canv_bbox.top < window.innerHeight) {
                if(canv_bbox.bottom > 0) {
                    canv.draw.clear()
                    this.update_draggables(canv.drag)
                    if(typeof canv.function == "function") {
                        canv.function(canv)
                    }
                }
            }

        }
    }

    update_draggables(objects) {
        for(let obj in objects) {
            objects[obj].update()
        }
    }

    resize_canvas(canv) {
        let canvas = canv.canvas
        let width = canvas.parentElement.getBoundingClientRect().width * canv.width

        canvas.width = width * this.scale
        canvas.height = width * canv.ratio * this.scale

        canvas.style.width = `${width}px`
        // canvas.style.height = `${width * canv.ratio}px`

        let aspect_ratio = window.innerWidth / window.innerHeight
        canvas.height /= aspect_ratio/2
    }
}

function get_color(color) {
    color = Theme.get(color) || color || "#ffffff"
    return color
}

class CDraw {
    static max_radius = 10
    static arrow_ratio = 1/2
    constructor(canv, arrowhead_size=20) {
        this.canv = canv
        this.canvas = canv.canvas
        this.c = this.canv.c
        this.arrowhead_size = arrowhead_size

        this.margin = CDraw.max_radius * 1.5

        this.clear()
    }

    clear() {
        this.c.clearRect(0,0, this.canvas.width,this.canvas.height)
    }

    point_x(x) {
        let working_width = (this.canvas.width/2) - this.margin
        let perc = x / this.canv.coord_width
        let ord = perc * working_width
        return ord + (this.canvas.width/2)
    }
    point_y(y) {
        let working_height = (this.canvas.width/2) - this.margin
        let perc = -y / this.canv.coord_width
        let ord = perc * working_height
        return ord + (this.canvas.height/2)
    }
    point_vector(pt) {
        return new Point(this.point_x(pt.x), this.point_y(pt.y))
    }

    point(pt, color, radius=6, convert_cords=true) {
        this.c.fillStyle = get_color(color)
        this.c.beginPath()

        if(convert_cords) {
            pt = this.point_vector(pt)
        }

        this.c.arc(pt.x, pt.y, radius, 0, 2*Math.PI)
        this.c.fill()
    }

    rect(x1, y1, x2, y2, color="white", width=2) {
        x1 = this.point_x(x1)
        y1 = this.point_y(y1)
        x2 = this.point_x(x2)
        y2 = this.point_y(y2)

        this.c.strokeStyle = get_color(color)
        this.c.strokeRect(x1, y1,  x2-x1, y2-y1)
    }
    fillrect(x1, y1, x2, y2, color="white") {
        x1 = this.point_x(x1)
        y1 = this.point_y(y1)
        x2 = this.point_x(x2)
        y2 = this.point_y(y2)

        this.c.fillStyle = get_color(color)
        this.c.fillRect(x1, y1,  x2-x1, y2-y1)
    }

    text(x, y, text, color, convert_cords=true, size=48) {
        this.c.fillStyle = get_color(color)
        // this.c.font = "24px 'Segoe UI'"
        this.c.font = `${size}px MJXc-TeX-math-I,MJXc-TeX-math-Ix,MJXc-TeX-math-Iw`
        this.c.textAlign = "center"

        if(convert_cords) {
            x = this.point_x(x)
            y = this.point_y(y)
        }

        this.c.fillText(text, x, y)
    }

    line_label(p1, p2, color, label, label_position, convert_cords=true) {
        color = get_color(color)
        if(convert_cords) {
            p1 = this.point_vector(p1)
            p2 = this.point_vector(p2)
        }

        let delta = p2.sub(p1)
        let center_delta = delta.scale(.5)
        let perp = delta.perpendicular().unit()
        let is_down = center_delta.y < 0
        let pos_multiplier = is_down? 1: -1

        let scale = Math.min(delta.length(), this.arrowhead_size)
        let label_dist_scale = scale / this.arrowhead_size

        let center_pos = p1.add(center_delta)
        let label_pos = center_pos.add(perp.scale(label_position * pos_multiplier * label_dist_scale))
        
        this.text(
            label_pos.x, 
            label_pos.y, 
            label, color, false,
            Math.min(delta.length(), 48)
        )
    }

    line(p1, p2, color, width=2, label="", label_pos=40, convert_cords=true) {
        this.c.strokeStyle = get_color(color)

        this.line_label(p1, p2, color, label, label_pos)
        if(convert_cords) {
            p1 = this.point_vector(p1)
            p2 = this.point_vector(p2)
        }

        this.c.lineWidth = width
        this.c.beginPath()
        this.c.moveTo(p1.x, p1.y)
        this.c.lineTo(p2.x, p2.y)
        this.c.stroke()
    }
    dottedline(p1, p2, color, width=2, label="", label_pos=40, convert_cords=true) {
        this.c.strokeStyle = get_color(color)
        this.line_label(p1, p2, color, label, label_pos)

        if(convert_cords) {
            p1 = this.point_vector(p1)
            p2 = this.point_vector(p2)
        }

        let delta = p2.sub(p1)
        let dir = delta.unit()
        let seg_length = 10
        let seg_num = delta.length() / seg_length

        for(let i = 1; i < seg_num; i += 2) {
            let start = p1.add(dir.scale(seg_length * (i-1)))
            let end   = p1.add(dir.scale(seg_length * i))

            this.line(start, end, color, width, "", 0, false)
        }
    }

    vector(p1, p2, color="white", label="", label_position=40, width=4) {
        color = get_color(color)
        this.line_label(p1, p2, color, label, label_position, true)
        
        p1 = this.point_vector(p1)
        p2 = this.point_vector(p2)
        
        // draw head
        let delta = p2.sub(p1)
        let perp = delta.perpendicular().unit()
        
        let scale = Math.min(delta.length(), this.arrowhead_size)
        delta = delta.unit()
        
        let perp_scale = perp.scale(scale * CDraw.arrow_ratio)
        let base_pos = p2.sub(delta.scale(scale))
        
        let corner1 = base_pos.add(perp_scale)
        let corner2 = base_pos.sub(perp_scale)
        
        this.line(p1, base_pos, color, width, "", 0, false)
        this.polygon([p2, corner1, corner2], color)
    }

    angle_between(a, b, origin, radius, color, label="θ", label_pos=40) {
        let a_angle = -Math.atan2(a.y, a.x)
        let b_angle = -Math.atan2(b.y, b.x)
        
        origin = this.point_vector(origin)
        a      = this.point_vector(a)
        b      = this.point_vector(b)

        // scale the markings
        let a_scale = a.sub(origin).length()
        let b_scale = b.sub(origin).length()
        let new_radius = Math.min(radius, a_scale * .5, b_scale * .5)
        let scale = new_radius / radius

        radius = new_radius
        label_pos *= scale

        // draw the segment
        this.c.strokeStyle = get_color(color)
        this.c.beginPath()

        let diff = a_angle - b_angle
        if(diff < 0) diff += 2* Math.PI

        // flip so the angle is on the acute side
        let center_angle = a_angle - .5*diff
        if(diff < Math.PI) {
            this.c.arc(origin.x, origin.y, radius, b_angle, a_angle)
        } else {
            this.c.arc(origin.x, origin.y, radius, a_angle, b_angle)
            center_angle += Math.PI
        }

        this.c.stroke()

        this.text(
            origin.x + (radius + label_pos) * Math.cos(center_angle),
            origin.y + (radius + label_pos) * Math.sin(center_angle),
            label, color, false,
            48 * scale
        )
    }

    polygon(points, color) {
        this.c.fillStyle = get_color(color)
        this.c.beginPath()

            this.c.moveTo(points[0].x, points[0].y)

            for(let i = 1; i < points.length; i++) {
                this.c.lineTo(points[i].x, points[i].y)
            }

        this.c.fill()
    }

    parametric_curve(func, start, stop, color="white", width=4, precision=.03) {
        color = get_color(color)

        // must draw the line in a single stroke to connect them
        this.c.strokeStyle = color
        this.c.lineWidth = width
        this.c.beginPath()

        for(let t = start; t < stop; t += precision) {
            t = Math.min(stop - precision, t) // don't overshoot the end
            let p1 = this.point_vector(func(t))
            let p2 = this.point_vector(func(t + precision))

            this.c.moveTo(p1.x, p1.y)
            this.c.lineTo(p2.x, p2.y)
        }

        this.c.stroke()
    }

    graph(func, start, stop, color="white", width=4, precision=.05) {
        this.parametric_curve(t => {
            return new Point(t, func(t))
        }, start, stop, color, width, precision)
    }

    axes(origin=Origin, x=true, y=true) {
        let center_rad = .1
        let edge_buffer = .6

        let xgradient = this.c.createRadialGradient(
            this.canvas.width/2, this.canvas.height/2, this.canvas.width * center_rad,
            this.canvas.width/2, this.canvas.height/2, this.canvas.width * edge_buffer
        )
        xgradient.addColorStop(0, "#888")
        xgradient.addColorStop(1, "#00000000")

        let ygradient = this.c.createRadialGradient(
            this.canvas.width/2, this.canvas.height/2, this.canvas.width * center_rad,
            this.canvas.width/2, this.canvas.height/2, this.canvas.width * edge_buffer
        )
        ygradient.addColorStop(0, "#888")
        ygradient.addColorStop(1, "#00000000")

        let right_edge  = new Point(-this.canv.coord_width, origin.y)
        let left_edge   = new Point( this.canv.coord_width, origin.y)
        let bottom_edge = new Point(origin.x, -this.canv.coord_width)
        let top_edge    = new Point(origin.x,  this.canv.coord_width)

        if(x) this.line(right_edge, left_edge, xgradient)
        if(y) this.line(bottom_edge, top_edge, ygradient)
    }
}


class Point {
    constructor(x, y) {
        this.x = x 
        this.y = y
    }

    x_comp() { return new Point(this.x, 0) }
    y_comp() { return new Point(0, this.y) }

    add(other) {
        return new Point(this.x + other.x, this.y + other.y)
    }
    sub(other) {
        return new Point(this.x - other.x, this.y - other.y)
    }
    scale(val) {
        return new Point(this.x * val, this.y * val)
    }
    length() {
        return Math.hypot(this.x, this.y)
    }
    elem_div(other) {
        return new Point(this.x / other.x, this.y / other.y)
    }
    elem_mult(other) {
        return new Point(this.x * other.x, this.y * other.y)
    }

    dist(other) {
        return other.sub(this).length()
    }

    unit() {
        let len = this.length()
        return new Point(this.x / len, this.y / len)
    }
    perpendicular() {
        return new Point(-this.y, this.x)
    }

    dot(other) {
        return this.x * other.x + this.y * other.y
    }

    toString() {
        return `Point(${this.x}, ${this.y})`
    }
}
const dot = (a, b) => {
    return a.dot(b)
}


class Draggable extends Point {
    static anim_time = .25
    static max_radius = 15
    static min_radius = 9
    static active_dist = .2
    constructor(x, y, name, canv, color, offset=Origin) {
        super(x, y)
        this.canv = canv
        this.color = get_color(color)
        this.active = false
        this.name = name
        this.radius = Draggable.max_radius
        this.offset = offset

        this.ease_in = false 
        this.ease_out = true

        canv.drag[name] = this
    }

    update() {
        let mouse = this.canv.mouse()
        if(this.active) {
            this.x = mouse.x - this.offset.x
            this.y = mouse.y - this.offset.y

            this.x = clamp(
                this.x, 
                -this.canv.coord_width - this.offset.x, 
                 this.canv.coord_width - this.offset.x
            )
            this.y = clamp(
                this.y, 
                 -this.canv.coord_width * .5 - this.offset.y, 
                  this.canv.coord_width * .5 - this.offset.y
            )
        }

        let vpos = this.add(this.offset)
        this.canv.draw.point(vpos, this.color + "70", this.radius)

        if(vpos.dist(mouse) < Draggable.active_dist) {
            
            if(!this.ease_in && !this.canv.dragging) {
                this.ease_in = true
                this.ease_out = false
                this.radius = new AnimatedValue(Draggable.max_radius, Draggable.min_radius, Draggable.anim_time)
            }  
            if(multicanv.mouse_down && !this.canv.dragging) {
                this.active = true
                this.canv.dragging = true
            }
        } else {
            if(!this.ease_out) {
                this.ease_out = true
                this.ease_in = false
                this.radius = new AnimatedValue(Draggable.min_radius, Draggable.max_radius, Draggable.anim_time)
            }
        }

        if(!multicanv.mouse_down) {
            this.active = false
            this.canv.dragging = false
        }
    }
}

const Origin = new Point(0, 0)

class AnimatedValue {
    constructor(current_value, target, anim_time, easing="smoothstep") {
        // stop nested animations causing rediculous performance hits
        if(current_value instanceof AnimatedValue) current_value = current_value.valueOf()
        if(target instanceof AnimatedValue) target = target.valueOf()

        this.easing = AnimatedValue[easing]
        this.anim_time = anim_time
        this.current_value = current_value
        this.target = target
        this.anim_start = time()
    }

    valueOf() {
        return this.value()
    }
    toString() {
        return round(this.value()).toString()
    }

    lerp(start, end, perc) {
        return start + (end - start) * perc
    }

    static linear(x) {
        return clamp(x, 0, 1)
    }

    static smoothstep(x) {
        if(x < 0) return 0
        if(x > 1) return 1
        return 3*x**2 - 2*x**3
    }

    value() {
        let dtime = time() - this.anim_start
        return this.lerp(
            this.current_value, this.target, 
            this.easing(dtime / this.anim_time)
        )
    }
}

class NumControler extends AnimatedValue {
    constructor(element, anim_time=.5, easing="smoothstep") {
        super(
            parseFloat(element.value), 
            parseFloat(element.value), 
            anim_time, easing
        )
        this.element = element

        element.addEventListener("input", () => {
            this.current_value = this.target
            this.target = parseFloat(this.element.value)
            this.anim_start = time()
        })
    }

    value() {
        let dtime = time() - this.anim_start
        let diff = this.target - this.current_value

        // snap to value if scrubbing slider
        if(this.element.type == "range") {
            if(Math.abs(diff) <= 2*parseFloat(this.element.step)) {
                return this.target
            }
        }

        return this.lerp(
            this.current_value, this.target, 
            this.easing(dtime / this.anim_time)
        )
    }
}

let multicanv = new MultiCanv()
window.addEventListener("load", () => {
    multicanv.autosetup()

    // run an init funciton if defined
    if(typeof window["init"] == "function") {
        init()
    }
})

  
