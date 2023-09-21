class Complex extends Point {
    static i = new Complex(0, 1)
    static zero = new Complex(0)

    constructor(real, imag=0) {
        super(real, imag)
        this.real = real
        this.imag = imag
    }

    round() {
        return new Complex(
            Math.round(this.real * 10) / 10,
            Math.round(this.imag * 10) / 10
        )
    }

    print() {
        console.log(this.toString())
    }

    toString() {
        let out = ""

        let printimag = this.imag
        if(this.imag ==  1) printimag = ""
        if(this.imag == -1) printimag = "-"

        // Purely Real/Imaginary Cases
        if(this.imag == 0) {
            out = this.real
        }else if(this.real == 0) {
            out = printimag + "i"
        } else {
            // Complex Number Case
            let parts = [
                this.real, 
                (Math.abs(this.imag)==1? "": Math.abs(this.imag)) + "i"
            ]
            let delim = this.imag > 0? " + ": " - "
            out = parts.join(delim)
        }
        return out
    }

    angle() {
        return Math.atan2(this.imag, this.real)
    }

    static complexify(n) {
        if(n instanceof Complex) return n
        if(n instanceof Point)   return new Complex(n.x, n.y)
        return new Complex(n)
    }

    add(other) {
        other = Complex.complexify(other)
        return new Complex(this.real + other.real, this.imag + other.imag)
    }

    sub(b) {
        b = Complex.complexify(b)
        return this.add(b.mult(-1))
    }

    mult(b) {
        b = Complex.complexify(b)
        return new Complex(
            this.real * b.real - this.imag * b.imag,
            this.real * b.imag + this.imag * b.real
        )
    }

    div(b) {
        b = Complex.complexify(b)

        let denominator = b.real ** 2 + b.imag ** 2
        return new Complex(
            (this.real * b.real + this.imag * b.imag) / denominator,
            (this.imag * b.real - this.real * b.imag) / denominator
        )
    }

    pow(power) {
        power = Complex.complexify(power)
        
        if(power.imag == 0) {
            power = power.real

            let sign = Math.sign(power)
            power = Math.abs(power)
    
            let prod = this
            for(let i = 0; i < power-1; i++) {
                prod = prod.mult(this)
            }
    
            if(sign == 1)  return prod
            if(sign == -1) return new Complex(1).divide(prod)
        } else {
            throw "Complex base not implemented for complex power."
        }

    }

    static exp(imaginary) { // expects imaginary component
        return new Complex(
            Math.cos(imaginary),
            Math.sin(imaginary)
        )
    }

    abs() {
        return this.length()
    }

    static intergrate(func, lowBound, highBound, precision=50) {
        // intergrates the function over real inputs
        // Precision is sample points per unit
        let domain = highBound - lowBound
        precision *= domain
    
        let sum = Complex.zero
        for(let x = lowBound; x < highBound; x += domain / precision) {
            sum = Complex.add(sum, func(x))
        }
        return sum
    }
}

var disp_num = new Complex(1)
var real_angle = disp_num.angle()
var real_length = disp_num.abs()
function init_real_numbers(canv) {
    canv.ratio = 1/4
    canv.buttons.negate.addEventListener("click", () => {
        disp_num = disp_num.mult(-1)

        let angle = disp_num.angle()
        if(real_angle == Math.PI) angle = 2*Math.PI
        if(real_angle == Math.PI*2) real_angle = 0

        real_angle = new AnimatedValue(real_angle, angle, .4)
    })
    canv.buttons.add.addEventListener("click", () => {
        disp_num = disp_num.add(1)
        if(disp_num.length() != 0) real_angle = disp_num.angle()
        real_length = new AnimatedValue(real_length, disp_num.length(), .4)
    })
    canv.buttons.sub.addEventListener("click", () => {
        disp_num = disp_num.sub(1)
        real_angle = disp_num.angle()
        real_length = new AnimatedValue(real_length, disp_num.length(), .4)
    })
}
function real_numbers(canv) {
    canv.draw.axes(Origin, true, false)

    let z = Complex.exp(real_angle).mult(real_length)
    canv.draw.vector(Origin, z, "blue", z.round().real, 50)

    if(z.length() == 0) {
        canv.draw.point(Origin, "blue")
        canv.draw.text(0, .2, "0", "blue")
    }
}
