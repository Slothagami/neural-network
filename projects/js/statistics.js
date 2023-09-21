var data
function init() {
    data = document.getElementById("data")
    random_data(30)
}

function norm(x, range = 2) {
    // input 0-1, returns gaussian value
    x = 2 * (x - .5) // scale -1 to 1
    x *= range
    return Math.exp(-(x**2)) * 100
}

function get_data() {
    samples = data.value.split(",")
    samples = samples.filter(x => /[0-9]/.test(x))
    samples = samples.map(x => parseFloat(x.trim()))
    return samples
}

function random_data(n_samples) {
    samples = []
    for(let i = 0; i < n_samples; i++) {
        samples.push(Math.round(norm(i/n_samples)))
    }
    data.value = samples.join(", ")
}

function order_data() {
    // orders the largest values in the middle
    let dat = get_data()
    dat.sort((a, b) => a - b) // sort ascending
    
    let disp = []
    for(let i = 0; i < dat.length; i += 2) {
        disp.push(dat[i])
    }
    let start = dat.length%2==0? dat.length-1: dat.length-2
    for(let i = start; i > 0; i -= 2) {
        disp.push(dat[i])
    }

    return disp
}

var width = 2.3
var line_y = -.5
function draw_slice(canv, dat, i) {
    let col_width = width/dat.length
    let col_start = -width/2 + col_width * i
    let height = dat[i] / Math.max(...dat)
    canv.draw.fillrect(
        col_start, line_y, 
        col_start + col_width, height + line_y, 
        "blue"
    )
    return [
        new Point(col_start, line_y),
        new Point(col_start + col_width, height + line_y)
    ]
}

function init_visualization(canv) {
    canv.coord_width = 1.5
}
function visualization(canv) {
    canv.draw.line(
        new Point(-width/2, line_y),
        new Point( width/2, line_y),
        "white"
    )

    // draw the rectangles
    let dat = order_data()
    for(let i = 0; i < dat.length; i++) {
        draw_slice(canv, dat, i)
    }
}
