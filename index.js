console.log("in index.js");

let models = {};
let noc2011_4d = {};
let soc2010_6d = {};


(async () => {
    console.log("loading noc2011...")
    noc2011_4d.occupations = await (await fetch("https://danielruss.github.io/codingsystems/noc_2011.json")).json()
    noc2011_4d.occupations = noc2011_4d.occupations.filter(code => code.level=="4")
    noc2011_4d.codes = noc2011_4d.occupations.map(code => code.code)
    noc2011_4d.code_lookup=noc2011_4d.occupations.reduce((pv,cv,indx)=>{
        pv[cv.code] = indx
        return pv;
    },{})
    noc2011_4d.title_lookup=noc2011_4d.occupations.reduce((pv,cv,indx)=>{
        pv[cv.code] = cv.title
        return pv;
    },{})
    console.log("noc2011 loaded...")
    console.log("loading soc2010...")
    soc2010_6d.occupations = (await (await fetch("https://danielruss.github.io/codingsystems/soc_2010_complete.json")).json()).codes
    Object.keys(soc2010_6d.occupations).forEach( (key) => {if(soc2010_6d.occupations[key].children) delete soc2010_6d.occupations[key]} )
    delete soc2010_6d.occupations['99-9999']
    soc2010_6d.codes = Object.keys(soc2010_6d.occupations)
    soc2010_6d.code_lookup=soc2010_6d.codes.reduce((pv,cv,indx)=>{
        pv[cv] = indx
        return pv;
    },{})
    soc2010_6d.title_lookup=Object.values(soc2010_6d.occupations).reduce((pv,cv)=>{
        pv[cv.code] = cv.title
        return pv;
    },{})
    console.log("soc2010 loaded...")

    console.log("loading models...")
    models.noc2011_encoder = await tf.loadLayersModel('./models/noc2011_4d_encoder_js/model.json')
    models.noc2011_decoder = await tf.loadLayersModel('./models/noc2011_4d_decoder_js/model.json')
    models.soc2010_encoder = await tf.loadLayersModel('./models/soc2010_6d_encoder_js/model.json')
    models.soc2010_decoder = await tf.loadLayersModel('./models/soc2010_6d_decoder_js/model.json')
    console.log("all models loaded...")
})()

async function noc_encode(code){
    if (!noc2011_4d.codes.includes(code)){
        console.log(`bad noc2011 code: ${code}`)
        return null
    }
    let indx = noc2011_4d.code_lookup[code]
    let buffer = tf.buffer([1,500]);
    buffer.set(1,0,indx);

    let res = models.noc2011_encoder.predict(buffer.toTensor())
    res.print()

    return res;
}

async function soc2010_encode(code){
    if (!soc2010_6d.codes.includes(code)){
        console.log(`bad soc2010 code: ${code}`)
        return null
    }
    let indx = soc2010_6d.code_lookup[code]
    let buffer = tf.buffer([1,840]);
    buffer.set(1,0,indx);

    let res = models.soc2010_encoder.predict(buffer.toTensor())
    res.print()

    return res;
}

async function noc_decode(enc){
    let res = models.noc2011_decoder.predict(enc);
    let indx_t = (await tf.whereAsync(res.greater(0.6),1,0));
    let indx = await indx_t.data();
    let codes = indx.reduce( (pv,cv,indx) =>{
        if (indx % 2 == 1){
            pv.push(noc2011_4d.codes[cv]);
        }
        return(pv)
    },[] )
    indx_t.dispose();
    return codes;
}

async function soc_decode(enc){
    let res = models.soc2010_decoder.predict(enc);
    res.print()
    let indx_t = (await tf.whereAsync(res.greater(0.6),1,0));
    let indx = await indx_t.data();
    let codes = indx.reduce( (pv,cv,indx) =>{
        if (indx % 2 == 1){
            pv.push(soc2010_6d.codes[cv]);
        }
        return(pv)
    },[] )
    indx_t.dispose();
    return codes;
}
