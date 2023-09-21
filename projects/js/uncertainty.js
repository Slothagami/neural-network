function integrate(func, a, b, prec) {
    let dx = (b-a)/prec
    let sum = 0
    for(let x = a; x < b; x += dx) {
        sum += func(x) * dx
    }
    return sum
}

function norm(x) {
    let s = spread**2
    return Math.exp(-.5 * (x**2) / s)
}
function norm_scale() {
    let s = spread**2
    return 1/(s * Math.sqrt(2 * Math.PI))
}

function fourier_transform_real(r, func, a, b, prec) {
    return integrate(x => func(x) * Math.cos(x * r), a, b, prec)
}
function fourier_transform_imag(r, func, a, b, prec) {
    return integrate(x => func(x) * Math.sin(x * r), a, b, prec)

}

function init_demo(canv){
    canv.coord_width = 8
}

var bound = 4
var freq = 4
var spread = 1
var position_real = x => {
    return Math.cos(x * freq) * norm(x)
}
var momentum_real = x => {
    return fourier_transform_real(x, position_real, -bound, bound, 40) / (2*bound) * 4// rescaled fourier transform
}

var position = x => {
    return norm(x) * norm_scale()
}
var momentum = x => {
    let real = fourier_transform_real(x, position_real, -bound, bound, 40)
    let imag = fourier_transform_imag(x, position_real, -bound, bound, 40)
    return Math.sqrt(real**2 + imag**2)
}

function demo(canv) {
    spread = canv.controls.spread
    let draw_bound = canv.coord_width

    // draw position amplitude function
    canv.draw.axes(Origin, true, false)
    canv.draw.graph(position, -draw_bound, draw_bound, "blue", 4, .03)
    canv.draw.graph(momentum, -draw_bound, draw_bound, Theme.get("orange") + "80", 4, .07)
}

function init_real_part(canv){
    canv.coord_width = 8
}
function real_part(canv) {
    spread = canv.controls.spread
    let draw_bound = canv.coord_width

    // draw position amplitude function
    canv.draw.axes(Origin, true, false)
    canv.draw.graph(position_real, -draw_bound, draw_bound, "blue", 4, .03)
    canv.draw.graph(momentum_real, -draw_bound, draw_bound, Theme.get("orange") + "80", 4, .07)
}




var wave_bound = 1.8
var line_x = -wave_bound
function init_relative_freq(canv) {
    let slider = canv.controls.speed.element
    slider.addEventListener("change", () => {
        line_x = new AnimatedValue(wave_bound, -wave_bound, slider.value, "linear")
    })
}
function relative_freq(canv) {
    let ysep = 1.5
    let wave = x => .5*(Math.cos(x * 16) * norm(x * 2) + ysep)
    canv.draw.graph(wave, -wave_bound, wave_bound, Theme.get("blue") + "80", 4, .01)

    // stationary wave
    canv.draw.line(
        new Point(line_x, ysep/2 + .5),
        new Point(line_x, ysep/2 - .5),
        "gray-1"
    )
    canv.draw.point(
        new Point(line_x, wave(line_x)),
        "white", 10
    )

    // relative wave
    let speed = canv.controls.speed.target
    canv.draw.line(
        new Point(line_x*speed, -ysep/2 + .5),
        new Point(line_x*speed, -ysep/2 - .5),
        "gray-1"
    )
    canv.draw.point(
        new Point(line_x*speed, wave(line_x) - ysep),
        "white", 10
    )

    canv.draw.graph(x => wave(x/speed)-ysep, line_x * speed, wave_bound*speed, Theme.get("orange"), 4, .01)
    canv.draw.vector(
        new Point(wave_bound*speed, -1.4),
        new Point(line_x * speed,   -1.4),
        "gray-3", "time spent walking"
    )
    canv.draw.vector(
        new Point(wave_bound, .1),
        new Point(line_x,     .1),
        "gray-3", "distance walked"
    )

}
