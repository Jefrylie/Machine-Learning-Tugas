% Fakta tentang jenis kelamin
pria(kakek_ayah).
pria(kakek_ibu).
pria(joni).
pria(jefry).

wanita(nenek_ayah).
wanita(nenek_ibu).
wanita(ana).

% Fakta hubungan keluarga
anak(joni, kakek_ayah).
anak(joni, nenek_ayah).
anak(ana, kakek_ibu).
anak(ana, nenek_ibu).

anak(jefry, joni).
anak(jefry, ana).

% Definisi aturan untuk relasi keluarga
kakek(X, Y) :- pria(X), anak(Z, X), anak(Y, Z).
nenek(X, Y) :- wanita(X), anak(Z, X), anak(Y, Z).
ayah(X, Y) :- pria(X), anak(Y, X).
ibu(X, Y) :- wanita(X), anak(Y, X).
