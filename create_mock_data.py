import os
import pandas as pd

def create_hindi_mock():
    records = [
        {
            "query_id": 1001,
            "source_lang": "eng_Latn",
            "target_lang": "hin_Devn",
            "query_type": "description",
            "Eng_Query": "where is the taj mahal?",
            "query": "ताजमहल कहाँ स्थित है?",
            "Eng_Answer": "The Taj Mahal is located in Agra, Uttar Pradesh, India, on the south bank of the Yamuna river.",
            "Answer": "ताजमहल भारत के उत्तर प्रदेश के आगरा शहर में यमुना नदी के दक्षिण तट पर स्थित है।",
            "passages": {
                "English_passages": [
                    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
                    "The Taj Mahal is an ivory-white marble mausoleum on the south bank of the Yamuna river in the Indian city of Agra. It was commissioned in 1632 by the Mughal emperor Shah Jahan.",
                    "Qutub Minar is a minaret and 'victory tower' that forms part of the Qutb complex, a UNESCO World Heritage Site in the Mehrauli area of New Delhi, India.",
                    "Red Fort is a historic fort in the Old Delhi neighbourhood of Delhi, India, that served as the main residence of the Mughal Emperors."
                ],
                "Translated_passages": [
                    "एफिल टॉवर फ्रांस के पेरिस में चैंप डी मार्स पर एक गढ़ा-लोहा जाली टॉवर है।",
                    "ताजमहल भारतीय शहर आगरा में यमुना नदी के दक्षिण तट पर एक हाथीदांत-सफेद संगमरमर का मकबरा है। इसे 1632 में मुगल सम्राट शाहजहाँ द्वारा कमीशन किया गया था।",
                    "कुतुब मीनार एक मीनार और 'विजय स्तंभ' है जो भारत के नई दिल्ली के महरौली क्षेत्र में यूनेस्को की विश्व धरोहर स्थल कुतुब परिसर का हिस्सा है।",
                    "लाल किला दिल्ली, भारत के पुरानी दिल्ली इलाके में एक ऐतिहासिक किला है, जो मुगल सम्राटों के मुख्य निवास के रूप में कार्य करता था।"
                ],
                "is_selected": [0, 1, 0, 0]
            },
            "meta": {
                "frequency_penalty": 0,
                "max_tokens": 1024,
                "model_name": "mock-translator",
                "presence_penalty": 0,
                "temperature": 0,
                "top_p": 1
            }
        },
        {
            "query_id": 1002,
            "source_lang": "eng_Latn",
            "target_lang": "hin_Devn",
            "query_type": "entity",
            "Eng_Query": "what is the capital of india?",
            "query": "भारत की राजधानी क्या है?",
            "Eng_Answer": "New Delhi is the capital of India.",
            "Answer": "नई दिल्ली भारत की राजधानी है।",
            "passages": {
                "English_passages": [
                    "Mumbai is the financial capital of India and the capital of Maharashtra state.",
                    "Kolkata is the capital of the Indian state of West Bengal. Located on the east bank of the Hooghly River.",
                    "New Delhi is the capital of India and an administrative district of the National Capital Territory of Delhi.",
                    "Chennai is the capital of the Indian state of Tamil Nadu. Located on the Coromandel Coast off the Bay of Bengal."
                ],
                "Translated_passages": [
                    "मुंबई भारत की वित्तीय राजधानी और महाराष्ट्र राज्य की राजधानी है।",
                    "कोलकाता भारतीय राज्य पश्चिम बंगाल की राजधानी है। हुगली नदी के पूर्वी तट पर स्थित है।",
                    "नई दिल्ली भारत की राजधानी है और दिल्ली के राष्ट्रीय राजधानी क्षेत्र का एक प्रशासनिक जिला है।",
                    "चेन्नई भारतीय राज्य तमिलनाडु की राजधानी है। बंगाल की खाड़ी के कोरोमंडल तट पर स्थित है।"
                ],
                "is_selected": [0, 0, 1, 0]
            },
            "meta": {
                "frequency_penalty": 0,
                "max_tokens": 1024,
                "model_name": "mock-translator",
                "presence_penalty": 0,
                "temperature": 0,
                "top_p": 1
            }
        },
        {
            "query_id": 1003,
            "source_lang": "eng_Latn",
            "target_lang": "hin_Devn",
            "query_type": "description",
            "Eng_Query": "what is photosynthesis?",
            "query": "प्रकाश संश्लेषण क्या है?",
            "Eng_Answer": "Photosynthesis is the process used by plants and other organisms to convert light energy into chemical energy.",
            "Answer": "प्रकाश संश्लेषण पौधों और अन्य जीवों द्वारा प्रकाश ऊर्जा को रासायनिक ऊर्जा में बदलने के लिए उपयोग की जाने वाली प्रक्रिया है।",
            "passages": {
                "English_passages": [
                    "Respiration is the movement of oxygen from the outside environment to the cells within tissues, and the removal of carbon dioxide.",
                    "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that, through cellular respiration, can later be released to fuel the organisms' activities.",
                    "Evaporation is a type of vaporization that occurs on the surface of a liquid as it changes into the gas phase.",
                    "Osmosis is the spontaneous net movement of solvent molecules through a selectively permeable membrane into a region of higher solute concentration."
                ],
                "Translated_passages": [
                    "श्वसन बाहरी वातावरण से ऊतकों के भीतर कोशिकाओं में ऑक्सीजन की गति और कार्बन डाइऑक्साइड को हटाने की प्रक्रिया है।",
                    "प्रकाश संश्लेषण पौधों और अन्य जीवों द्वारा प्रकाश ऊर्जा को रासायनिक ऊर्जा में परिवर्तित करने के लिए उपयोग की जाने वाली एक प्रक्रिया है, जिसे बाद में सेलुलर श्वसन के माध्यम से जीवों की गतिविधियों को बढ़ावा देने के लिए जारी किया जा सकता है।",
                    "वाष्पीकरण एक प्रकार का वाष्पीकरण है जो किसी तरल की सतह पर होता है जब वह गैस चरण में बदल जाता है।",
                    "परासरण चुनिंदा पारगम्य झिल्ली के माध्यम से विलायक अणुओं की उच्च विलेय सांद्रता वाले क्षेत्र में सहज शुद्ध गति है।"
                ],
                "is_selected": [0, 1, 0, 0]
            },
            "meta": {
                "frequency_penalty": 0,
                "max_tokens": 1024,
                "model_name": "mock-translator",
                "presence_penalty": 0,
                "temperature": 0,
                "top_p": 1
            }
        }
    ]
    df = pd.DataFrame(records)
    os.makedirs("data", exist_ok=True)
    df.to_parquet("data/hinval_mini.parquet", index=False)
    print("Created data/hinval_mini.parquet with 3 detailed records.")

def create_marathi_mock():
    records = [
        {
            "query_id": 2001,
            "source_lang": "eng_Latn",
            "target_lang": "mar_Devn",
            "query_type": "description",
            "Eng_Query": "where is the gateway of india?",
            "query": "गेटवे ऑफ इंडिया कुठे आहे?",
            "Eng_Answer": "The Gateway of India is located in Mumbai, Maharashtra, India, overlooking the Arabian Sea.",
            "Answer": "गेटवे ऑफ इंडिया हे भारतातील महाराष्ट्र राज्यातील मुंबई शहरात अरबी समुद्राला तोंड करून स्थित आहे.",
            "passages": {
                "English_passages": [
                    "The Taj Mahal is located in Agra on the banks of Yamuna river.",
                    "The Gateway of India is an arch-monument built in the early twentieth century in the city of Bombay (now Mumbai), India. It overlooks the Arabian Sea.",
                    "Charminar is a mosque and monument located in Hyderabad, Telangana, India.",
                    "Hawa Mahal is a palace in the city of Jaipur, India. Built from red and pink sandstone."
                ],
                "Translated_passages": [
                    "ताजमहल आग्रा येथे यमुना नदीच्या काठावर आहे.",
                    "गेटवे ऑफ इंडिया हे विसाव्या शतकाच्या सुरुवातीला बॉम्बे (आताचे मुंबई) शहरात बांधलेले एक कमान-स्मारक आहे. हे अरबी समुद्राला तोंड करून आहे.",
                    "चारमिनार ही भारत देशातील तेलंगणा राज्यातील हैदराबाद येथील मशिद आणि स्मारक आहे.",
                    "हवा महल हा भारतातील जयपूर शहरातील एक राजवाडा आहे. लाल आणि गुलाबी वाळूच्या खडकापासून बनवलेला आहे."
                ],
                "is_selected": [0, 1, 0, 0]
            },
            "meta": {
                "frequency_penalty": 0,
                "max_tokens": 1024,
                "model_name": "mock-translator",
                "presence_penalty": 0,
                "temperature": 0,
                "top_p": 1
            }
        },
        {
            "query_id": 2002,
            "source_lang": "eng_Latn",
            "target_lang": "mar_Devn",
            "query_type": "entity",
            "Eng_Query": "what is the capital of maharashtra?",
            "query": "महाराष्ट्राची राजधानी कोणती आहे?",
            "Eng_Answer": "Mumbai is the capital of Maharashtra.",
            "Answer": "मुंबई ही महाराष्ट्राची राजधानी आहे.",
            "passages": {
                "English_passages": [
                    "Pune is the second largest city in the Indian state of Maharashtra after Mumbai.",
                    "Nagpur is the winter capital of Maharashtra state and a major commercial center.",
                    "Mumbai is the capital city of the Indian state of Maharashtra. As of 2011, it was the most populous city in India.",
                    "Nashik is an ancient holy city in Maharashtra, located on the banks of Godavari river."
                ],
                "Translated_passages": [
                    "पुणे हे मुंबई नंतर भारतातील महाराष्ट्र राज्यातील दुसरे मोठे शहर आहे.",
                    "नागपूर ही महाराष्ट्र राज्याची हिवाळी राजधानी असून एक मोठे व्यापारी केंद्र आहे.",
                    "मुंबई ही भारतातील महाराष्ट्र राज्याची राजधानी आहे. २०११ पर्यंत, हे भारतातील सर्वाधिक लोकसंख्या असलेले शहर होते.",
                    "नाशिक हे महाराष्ट्रातील एक प्राचीन पवित्र शहर आहे, जे गोदावरी नदीच्या काठावर वसलेले आहे."
                ],
                "is_selected": [0, 0, 1, 0]
            },
            "meta": {
                "frequency_penalty": 0,
                "max_tokens": 1024,
                "model_name": "mock-translator",
                "presence_penalty": 0,
                "temperature": 0,
                "top_p": 1
            }
        }
    ]
    df = pd.DataFrame(records)
    os.makedirs("data", exist_ok=True)
    df.to_parquet("data/marval_mini.parquet", index=False)
    print("Created data/marval_mini.parquet with 2 detailed records.")

def main():
    create_hindi_mock()
    create_marathi_mock()

if __name__ == "__main__":
    main()
