The microservice to make inference for taxonomy prediction.

Run pip install -r requirements.txt


To start the service run uvicorn app.main:app.

Download the model from https://drive.google.com/drive/folders/10Wobf4QZsxC4VCwmXWsdvJ2RY09Z-1-S?usp=sharing

For complete list of APIs navigate to http://localhost:8000/docs

For the model inference API use the endpoint http://localhost:8000/gettaxonomy and submit a post request with the content as payload

Example:
curl -X 'POST' \
  'http://localhost:8000/gettaxonomy' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{  "content": "In this video we'\''ll see how to synthesize alcohols using the Grignard reagents. So first, we have to learn how to make a Grignard reagent. So you start with an alkyl halide, so over here on the left. And you add a magnesium metal. And you need to add something like diethyl ether as your solvent. You can'\''t have any water present because water will react with the Grignard reagent. And so this is what you make, over here on the right. You end up with a carbon atom bonded to a metal. Right? So carbon is bonded to magnesium. This is called an organometallic bond. And you can do this with other metals. You can do this with lithium, for example. But Grignard reagents are one of those things that'\''s always talked about in undergraduate organic chemistry classes. And you can see that these two electrons here, these red ones, the ones in red. I'\''ve drawn it like a covalent bond. The bond between carbon and magnesium. But in reality, it'\''s more ionic than covalent. So it'\''s equivalent to the second structure down here. Now, in terms of electronegativities, carbon is actually more electronegative than magnesium. So the two electrons in red are actually going to be closer to the carbon atom, itself, giving the carbon a negative charge, and forming a carbanion. And so this is a carbanion that is formed. And this is unique because this carbanion can now act as a nucleophile in your mechanism to make alcohols. So this is the preparation of a Grignard reagent, it'\''s proved to be a very, very useful thing in organic synthesis, so much so that Victor Grignard won the Nobel Prize for his research into this chemistry. Let'\''s take a look at the mechanism to form a Grignard reagent. So I'\''m going to start with my alkyl halide. And this time I'\''ll draw in all of my lone pairs on my halogen, like that. And we'\''re going to add magnesium, which we know, being in Group 2, magnesium has two valence electrons. I'\''m going to draw magnesium'\''s two valence electrons like that. In the first step of the mechanism, magnesium is going to donate one of its electrons. All right? So we'\''re going to show the movement of one of its electrons over here to this carbon. We'\''re going to use a half-headed arrow like that. So, we'\''re going to make a new anion here, because this carbon actually picks up an electron. So it actually picks up a negative charge. And we call this an anion radical, OK? So this intermediate here is called an anion radical. So I'\''ll go ahead and write that. So it'\''s an anion radical. It'\''s an anion because it picked up an electron, giving it a negative charge. And it'\''s radical because that electron is unpaired. So this anion radical is unstable and it'\''s going to fragment. So these two electrons right here are going to come off, onto the halogen. Right? So let'\''s go ahead and draw what that would give us. We now have this carbon with one electron around it on the right side, like that. And now, our halogen over here, it had three lone pairs. It just picked up another. So it is now negatively charged, like that. And the magnesium that we started with donated an electron.."
}'
