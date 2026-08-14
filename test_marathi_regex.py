import sys
import re

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

text = "एक कंपनी एका विशिष्ट देशात, बहुधा त्या देशाच्या लहान उपसमूहाच्या, जसे की राज्य किंवा प्रांताच्या सीमांतर्गत स्थापित केली जाते. नंतर कॉर्पोरेशन त्या राज्यातील समावेशाच्या कायद्यांद्वारे शासित केले जाते."

print("Original text:")
print(text)

# Tokenize WITHOUT word boundaries \b
words = re.findall(r'[a-zA-Z\u0900-\u097F]+', text.lower())
print("\nTokens without \\b:")
print(words)

# Check specific words
print("\nIs 'किंवा' in tokens?", "किंवा" in words)
print("Is 'केले' in tokens?", "केले" in words)
print("Is 'केली' in tokens?", "केली" in words)
