const data_a = [
    [-2.51589581, -0.96261202],
    [ 2.2990065 , -1.83511128],
    [-1.28902671, -1.11102898],
    [-1.21628901, -1.84443202],
    [-0.96443525, -0.34629715],
    [-1.02356894, -0.52868701],
    [-2.51940873, -0.04412912],
    [-2.32445006, -2.0303193 ]
]
const data_b = [
    [ 2.74316611,  0.48119561],
    [ 0.87961729,  2.14903472],
    [ 2.31882016,  1.28456482],
    [ 0.29951658,  1.40239238],
    [ 1.1438077 ,  1.25634619],
    [-0.97646496,  1.93671066],
    [ 2.32900555,  2.31877997],
    [ 0.12080216,  1.58382624]
]
const labels = [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1]

class SVM {
    constructor(eps=.1) {
        this.eps = eps
        this.randomize_weights()
    }

    randomize_weights() {
        this.weights = new Point(Math.random(), Math.random())
        this.bias    = Math.random()
    }

    fit(data, labels, lr=.001) {
        // performs one iteration of training (so that it trains more each frame)
        for(let i = 0; i < data.length; i++) {
            let sample = data[i]
            let label  = labels[i]

            let margin = label * (dot(this.weights, sample) - this.bias)
            if(margin >= 1) {
                this.weights = this.weights.sub(
                    this.weights.scale(lr * 2 / this.eps)
                )
            } else {
                this.weights = this.weights.sub(
                    this.weights.scale(2 / this.eps).sub(sample.scale(label)).scale(lr)
                )
                this.bias -= lr * label
            }
        }
    }

    predict(sample) {
        return Math.sign(dot(this.weights, sample) - this.bias)
    }

    draw(canv) {
        let slope     = -this.weights.x / this.weights.y
        let intercept = this.bias / this.weights.y

        let m1_intercept = (this.bias + 1) / this.weights.y
        let m2_intercept = (this.bias - 1) / this.weights.y

        canv.draw.graph(y => slope * y + m1_intercept, -8, 8, "gray-2")
        canv.draw.graph(y => slope * y + m2_intercept, -8, 8, "gray-2")
        canv.draw.graph(y => slope * y + intercept, -8, 8, "white")
    }
}

function init_demo(canv) {
    canv.coord_width = 5

    let n = 0
    data_a.forEach(point => {
        canv.add_draggable(point[0], point[1], n++, "orange")
    })

    data_b.forEach(point => {
        canv.add_draggable(point[0], point[1], n++, "blue")
    })

    canv.svm = new SVM()
}
function demo(canv) {
    // get data from the draggables
    let data = Object.values(canv.drag)
    canv.svm.eps = canv.controls.hardness ** 2

    data.forEach(dp => {
        canv.draw.point(dp, dp.color)
    })

    canv.svm.fit(data, labels)
    canv.svm.draw(canv)
}


function init_kernel_trick(canv) {
    canv.svm = new SVM(1e4)

    canv.data = []
    canv.data_labels = []
    canv.orange_test = point => Math.abs(point.x + .2) < .6

    for(let i = -3; i < 3; i += 1/5) {
        let pt = new Point(i, -1)
        canv.data.push(pt)
        canv.data_labels.push(canv.orange_test(pt)? 1: -1)
    }

    canv.buttons.apply.addEventListener("click", () => {
        canv.data.forEach(point => {
            point.y = new AnimatedValue(point.y, 2 * (point.x * .7) ** 2 - 1, .5)
        })
    })
}
function kernel_trick(canv) {
    canv.data.forEach(dp => {
        canv.draw.point(dp, canv.orange_test(dp)? "orange": "blue")
    })

    canv.svm.fit(canv.data, canv.data_labels, .002)
    canv.svm.draw(canv)
}


function init_hyperplane(canv) {
    canv.add_draggable(.5, 1.5, "w", "orange")
    canv.add_draggable(0, -.6, "b", "white")
}
function hyperplane(canv) {
    canv.drag.b.x = 0
    canv.drag.w.offset = canv.drag.b

    canv.draw.vector(
        canv.drag.b, 
        canv.drag.b.add(canv.drag.w), 
        "orange", "w"
    )
    canv.draw.dottedline(Origin, canv.drag.b, "gray-2", undefined, "b * w.y", -100)

    let b = canv.drag.b.y * canv.drag.w.y
    let slope     = -canv.drag.w.x / canv.drag.w.y
    let intercept =  b / canv.drag.w.y

    let m1_intercept = (b + 1) / canv.drag.w.y
    let m2_intercept = (b - 1) / canv.drag.w.y

    canv.draw.graph(y => slope * y + m1_intercept, -8, 8, "gray-2")
    canv.draw.graph(y => slope * y + m2_intercept, -8, 8, "gray-2")
    canv.draw.graph(y => slope * y + intercept, -8, 8, "white")
}
