
## Task 1 
> In order to prove that you got the system to run, paste the output for the following example sentence as
your solution to this task: In order to test the system, we feed it this sentence.


``` bash

'CONDITION -1 event.v.01 Participant +1 order.n.01 CONSEQUENCE -1 entity.n.01 test.v.01 Agent -1 Theme +1 system.n.01 person.n.01 EQU speaker feed.v.01 Participant -3 Agent -1 Time +1 Recipient +2 Time +3 time.n.08 EQU now entity.n.01 sentence.n.01'
```

## Task 2

> Paste your code for the function into your solution for this task

```python
from sre_parse import Tokenizer
from tokenization_mlm import MLMTokenizer
from transformers import MBartForConditionalGeneration

def parse_to_drs(input_str: str, tokenizer: Tokenizer) :
    model = MBartForConditionalGeneration.from_pretrained('laihuiyuan/DRS-LMM')
    inp_ids = tokenizer.encode(input_str,return_tensors="pt")
    foced_ids = tokenizer.encode("<drs>", add_special_tokens=False, return_tensors="pt")
    outs = model.generate(input_ids=inp_ids, forced_bos_token_id=foced_ids.item(), num_beams=5, max_length=150)
    text = tokenizer.decode(outs[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text

# i also added reverse function to translate drs to english, it will be used in later tasks
drs_tokenizer = MLMTokenizer.from_pretrained('laihuiyuan/DRS-LMM', src_lang='<drs>')

def parse_from_drs(input_str: str, tokenizer: Tokenizer):
     model = MBartForConditionalGeneration.from_pretrained('laihuiyuan/DRS-LMM')
     inp_ids = tokenizer.encode(input_str,return_tensors="pt")
     foced_ids = tokenizer.encode("en_XX", add_special_tokens=False, return_tensors="pt")
     outs = model.generate(input_ids=inp_ids, forced_bos_token_id=foced_ids.item(), num_beams=5, max_length=150)
     text = tokenizer.decode(outs[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
     return text
```

> show that it works by pasting the
output on the German sentence Mieter von Sozialwohnungen könnten nach Plänen der Bauministerin bald
zusätzliche Abgaben zahlen müssen, wenn ihre Bedürftigkeit wegfällt. 


```python

>>> de_tokenizer = MLMTokenizer.from_pretrained('laihuiyuan/DRS-LMM', src_lang='de_DE')
>>> parse_to_drs("Mieter von Sozialwohnungen könnten nach Plänen der Bauministerin bald zusätzliche Abgaben zahlen müssen, wenn ihre Bedürftigkeit wegfällt.", de_tokenizer)
'person.n.01 Role +1 Mieter.n.01 Theme +1 sozialwohnungen.n.01 könnten.v.01 Experiencer -3 Time +1 Stimulus +5 time.n.08 TPR now plan.n.01 bauministerin.n.01 additional.a.01 AttributeOf +1 contribution.n.01 Theme -2 pay.v.01 Agent -8 Theme -1 müssen.v.01 Agent -9 Time -6 Topic -1 person.n.01 need.n.01 Of -1 wegfällt.v.01 Time -10 Theme -1'
>>> 
```

> Inspect the output and check it against
(a translation of) the sentence - does it work as well as the Dutch example?

```python
>>> parse_to_drs("Ifølge planer fra byggeministeren kan lejere af almene boliger snart skulle betale yderligere gebyrer, hvis deres behov for almene boliger ophører.", nl_tokenizer)
'ølge.a.01 Theme +1 planer.n.01 fra.a.01 Theme -1 Location +1 byggeministeren.n.01 POSSIBILITY -1 lejere.v.01 Agent -3 Theme +1 almene_boliger.n.01 snart.v.01 Agent -1 Time +1 Theme +2 time.n.08 EQU now skulle.n.01 betale.v.01 Agent -5 Time +1 Theme +3 time.n.08 TPR now yderligere.a.01 Theme +1 gebyrer.n.01 Participant +1 person.n.'

```

To me Stimulus and Possibility are not the same thing. I'm not a native neither in German nor Dutch, may be there is no way to express stimulus in Dutch the same way as in German.
It also looks like German version has more drs relations, I guess it's because of case system. Both languages use compound words and in both cases drs doesn't split it to show the relation.

## Task 3

> Run your function with the English tokeniser on the following three example sentences

```python
>>> en_tokenizer = MLMTokenizer.from_pretrained('laihuiyuan/DRS-LMM', src_lang='en_XX')
>>> parse_to_drs("Whoever stole the money should be punished.", en_tokenizer)  # a
'NEGATION -1 person.n.01 steal.v.01 Agent -1 Time +1 Theme +2 time.n.08 TPR now money.n.01 NEGATION -1 NECESSITY -1 punish.v.01 Patient -4'
>>> parse_to_drs("He always listens to serious music.",  en_tokenizer)  # b
'NEGATION -1 time.n.08 NEGATION -1 male.n.02 listen.v.01 Time -2 Agent -1 Theme +3 time.n.08 EQU now EQU -3 serious.a.01 AttributeOf +1 music.n.01'
>>> parse_to_drs("I am not happy with my looks.", en_tokenizer)  # c
'person.n.01 EQU speaker NEGATION -1 time.n.08 EQU now happy.a.01 Experiencer -2 Time -1 Stimulus +1 look.n.01 Creator speaker'
```

> Interpret the results, and comment on their quality. 

in a) there shall be no negation, but drs pictures us a double negation. I don't understand to what first "-1" belongs to, because there is no previous node. the only explanation i came up with is that -1 means not an object but a box, the same very box that the existence of negation operator is introducing 

in b) double negation makes sense, because it literally means "there is **no** time when he does **not** listen to serious music"

in c) the representation is perfect except that "my" is expression of posession, not creation.

However I ran reverse function just for fun and it definetely compiles back to origin sentences

```python
>>> parse_from_drs(parse_to_drs("Whoever stole the money should be punished.", en_tokenizer), drs_tokenizer )                                                                                                   'Whoever stole the money should be punished.'
>>> parse_from_drs(parse_to_drs("He always listens to serious music.", en_tokenizer), drs_tokenizer )
'He always listens to serious music.'
>>> parse_from_drs(parse_to_drs("I am not happy with my looks.", en_tokenizer), drs_tokenizer )
"I'm not happy with my looks."
```

it means that even if drs string may not be perfect, it still represents meaning nicely 

## Task 4

> Run your function with the respective tokeniser on the
following parallel sentences

```python
>>> en_tokenizer = MLMTokenizer.from_pretrained('laihuiyuan/DRS-LMM', src_lang='en_XX')
>>> de_tokenizer = MLMTokenizer.from_pretrained('laihuiyuan/DRS-LMM', src_lang='de_DE')
>>> nl_tokenizer = MLMTokenizer.from_pretrained('laihuiyuan/DRS-LMM', src_lang='nl_XX')
>>> it_tokenizer = MLMTokenizer.from_pretrained('laihuiyuan/DRS-LMM', src_lang='it_IT')
>>> en_str = "I am ready to face any challenge."
>>> de_str = "Ich bin bereit, mich jeder Herausforderung zu stellen."
>>> nl_str = "Ik ben klaar om elke uitdaging aan te gaan."
>>> it_str = "Sono pronta ad affrontare qualsiasi sfida."
>>> parse_to_drs(en_str, en_tokenizer)
'person.n.01 EQU speaker time.n.08 EQU now ready.a.01 Experiencer -2 Time -1 Stimulus +1 face.v.01 Agent -3 Theme +1 challenge.n.01'
>>> parse_to_drs(de_str, de_tokenizer)
'entity.n.01 person.n.01 EQU speaker time.n.08 EQU now willing.a.01 Theme -2 Time -1 Participant +3 mich.n.01 challenge.n.01 stellen.v.01 Agent -6 Theme -1'
>>> parse_to_drs(nl_str, nl_tokenizer)
'person.n.01 EQU speaker time.n.08 EQU now ready.a.01 Experiencer -2 Time -1 Stimulus +1 challenge.n.01 entity.n.01 gaan.v.01 Theme -2 Agent -1'
>>> parse_to_drs(it_str, it_tokenizer)
'time.n.08 EQU now sono.n.01 ready.a.01 Time -2 Experiencer -1 Stimulus +1 face.v.01 Agent -2 Theme +1 challenge.n.01'
>>>
>>> en_str = "I usually keep a diary when I travel"
>>> de_str = "Normalerweise führe ich auf Reisen ein Tagebuch."
>>> nl_str = "Gewoonlijk hou ik een dagboek bij als ik op reis ga."
>>> it_str = "Durante i viaggi di solito tengo un diario."
>>> parse_to_drs(en_str, en_tokenizer)
'person.n.01 EQU speaker usually.a.01 keep.v.01 Agent -2 Manner -1 Time +1 Theme +2 time.n.08 EQU now diary.n.01 person.n.01 EQU speaker travel.v.01 TIN -4 Theme -1'
>>> parse_to_drs(de_str, de_tokenizer)
'führe.v.01 Time +1 Theme +2 Agent +3 Instrument +4 time.n.08 EQU now person.n.01 EQU speaker travel.n.01 diary.n.01'
>>> parse_to_drs(nl_str, nl_tokenizer)
'usually.r.01 keep.v.01 Manner -1 Time +1 Theme +2 Agent +3 time.n.08 EQU now person.n.01 EQU speaker diary.n.01 Theme +1 entity.n.01 person.n.01 EQU speaker trip.n.01 go.v.01 Time -6 Participant -3 Patient -2 Theme -1'
>>> parse_to_drs(it_str, it_tokenizer)
'travel.n.01 Theme +1 usually.n.01 keep.v.01 Agent -2 Time +1 Theme +2 time.n.08 EQU now diary.n.01'
>>> 
```

> How different are the outputs between the languages? 

In first example all drs's are pretty isomorphic to each other with word order regard.
In second one there is an obvious difference in amount of relations in different languages

> Based on the English versions, do the outputs for  other languages make sense?

I tried to run drs's from other languages back into English, but it seems that there are some untranslated words, which ruins the logic

```python
>>> lst = [
... (en_str, en_tokenizer),
... (de_str, de_tokenizer),
... (nl_str, nl_tokenizer),
... (it_str, it_tokenizer)
... ]

>>> for str, tokenizer in lst:
...     parse_from_drs(parse_to_drs(str, tokenizer), drs_tokenizer)
... 
"I'm ready to face the challenge."
"I'm willing to mich any challenge to stellen."
"I'm ready for any challenge it is to gaan."
"Now, I'm ready to face the challenge."

# and for the second part:
'I usually keep a diary when I travel.'
'Führe mir die Reisen in einem Tagebuch.'
"Usually I keep a diary of what I'm doing on trips."
'Travellers of usually keep a diary.'
```

The last italian sentence might be explained that the verb _tengo_ has no corresponding pronoun, and drs messed it up. In the fist italian example the sentence starts with _sono_ which must be a much more common thing, so it just has to occur in train data more often and ended in better result

> Are there any properties of the output which surprise you?

No, all of them we have seen in examples before

## Task 5

> parse the following typical newspaper sentence 

```python
>>> en_str = """The administration has pushed for the prosecution of the president’s political opponents, fired government
... employees for taking positions perceived as less than entirely loyal to Trump, and barred certain law firms
... from working with the government because they displeased the president."""
>>> 
>>> parse_to_drs(en_str, en_tokenizer)
'administration.n.01 push.v.01 Theme -1 Time +1 Destination +5 time.n.08 TPR now prosecution.n.01 PartOf +1 person.n.01 Role +1 president.n.01 political.a.01 AttributeOf +1 opponent.n.01 Creator -3 CONTINUATION -1 fire.v.01 Agent -8 Time -6 Theme +3 time.n.08 TPR now EQU -7 government.n.01 person.n.01 Theme -1 Role +1 employee.n.01 CONTINUATION -1 entity.n'
>>> 
>>> parse_from_drs(parse_to_drs(en_str, en_tokenizer), drs_tokenizer)
"The administration pushed for the prosecution of the president's political opponents, and fired the government employees. It was the first time that the president had resigned."
>>> 
```

> Analyse the output very closely, and comment on the quality compared to what you have seen in previous tasks.

Part of the sentence before first comma looks ok, but then it goes wrong. 
I guess there is a problem with Continuation statement. 
I tested it on simple example
```python
>>> en_str = "i like apples, bananas, cucumbers, dates, eggplants, guavas"
>>> parse_from_drs(parse_to_drs(en_str, en_tokenizer), drs_tokenizer)                                                                                                                                       
'I like apples, bananas, cucumbers, dates, eggplants and guava.'
```
but it worked fine.
Then I decided to transform homogenous elements into sentences
```python
>>> en_str = "i like to eat apples and I like to eat bananas and I like to eat cucumbers and i like to eat dates"
>>> parse_from_drs(parse_to_drs(en_str, en_tokenizer), drs_tokenizer)
'I like to eat apples and I like to eat bananas.'
```
and it proved me right
