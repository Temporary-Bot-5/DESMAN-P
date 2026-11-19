# DESMAN(P): poradnik dla uczestników testu

Dziękujemy za pomoc. Z tego poradnika dowiesz się, jak zainstalować DESMAN, co wypróbować i jak
zgłaszać problemy. Zacznij od rozdziałów 1–3, a do reszty zajrzyj w razie potrzeby.

## 0. Przeczytaj najpierw

- **To prototyp.** Może się zawiesić, zamknąć bez ostrzeżenia, dawać dziwne odpowiedzi lub tracić
  rozpoczętą pracę. Nie powierzaj mu ważnych spraw ani zadań, których nie można powtórzyć.
  Znalezione problemy są najważniejszym wynikiem testu. Zgłoś każdy z nich (rozdział 7).
- **Klucze zostają u ciebie.** Klucz dostępu do modelu (klucz API) jest zapisany tylko w pliku na
  twoim komputerze i tylko ty możesz go zobaczyć. Nie trafia do sieci DESMAN, właściciela, modelu
  ani historii rozmowy. Sieć dostaje tylko *odcisk* klucza, czyli krótki ciąg znaków, który
  odróżnia go od innych kluczy i nie pozwala go odtworzyć.
- **Tylko ty włączasz sieć.** Dopóki nie klikniesz **Dołącz** (rozdział 5), DESMAN w ogóle nie
  kontaktuje się z serwerem właściciela. Zadania, rozmowy i pliki zostają na twoim komputerze.
  Żądania trafiają tylko do dostawców modeli, których klucze dodasz.
- **Co operator (właściciel serwera) widzi po twoim dołączeniu do sieci:**
  - twój węzeł: odcisk jego klucza, czas ostatniego kontaktu, udostępnione modele i ich dzienne
    limity (dostawcę, model i odcisk klucza, bez samego klucza);
  - pracę wykonaną przez twój komputer dla sieci i jej wynik;
  - wysłane zgłoszenia problemów z dokładnie tym tekstem, który był w oknie potwierdzenia
    (rozdział 7);
  - w komunikatorze twoją nazwę oraz to, kto, do kogo i kiedy pisze. Wiadomości są szyfrowane i
    operator ich nie przeczyta, **chyba że wpłynie skarga**. Jeśli zgłosisz konkretną wiadomość,
    operator otrzyma jej treść;
  - adres IP, z którego nastąpiło połączenie, tak jak w przypadku każdej strony.
- **Uruchamianie kodu w systemie Windows.** Zainstaluj i uruchom Docker Desktop w trybie
  kontenerów linuksowych, a następnie wykonaj polecenie `desman sandbox setup --yes`, aby
  zatwierdzić pobranie obrazu. W ustawieniach DESMAN możesz włączyć uprawnienie do uruchamiania
  kodu (`shell_execution`). Bez piaskownicy DESMAN może napisać kod i poprosi o pomoc przy
  jego uruchomieniu.

## 1. Instalacja

Właściciel wyśle ci odnośnik do strony z przyciskami instalacji. Potrzebujesz dostępu do
internetu. Uprawnienia administratora są zbędne.

### Windows

1. Kliknij **Windows** na stronie, aby pobrać `install.bat`. Jeśli przeglądarka zapyta, czy
   zachować plik, wybierz „Zachowaj”.
2. Otwórz folder Pobrane i kliknij dwukrotnie `install.bat`. Jeśli pojawi się niebieskie okno
   „System Windows ochronił ten komputer”, wybierz „Więcej informacji” → „Uruchom mimo to”.
3. W czarnym oknie instalator zapyta o zgodę na instalację uv, Pythona i, jeśli go brakuje, Gita.
   Wpisz `y` i naciśnij Enter. To potrwa kilka minut.
4. Na koniec przeglądarka otworzy stronę DESMAN, a na pulpicie pojawi się skrót **DESMAN**. Od tej
   pory uruchamiaj program tym skrótem.

### Linux

1. Kliknij na stronie **Linux** i skopiuj wyświetlone polecenie.
2. Otwórz terminal, wklej polecenie (Ctrl+Shift+V) i naciśnij Enter.
3. Jeśli nie masz Gita, instalator pokaże polecenie do jego instalacji. Wykonaj je i powtórz krok
   2.
4. Na pytania odpowiadaj `y`. Na koniec otworzy się przeglądarka, a **DESMAN** pojawi się w menu
   aplikacji i na pulpicie. Jeśli pulpit zapyta o zgodę na uruchomienie skrótu, udziel jej.

Więcej szczegółów znajdziesz w [install.md](install.md).

## 2. Pierwsze uruchomienie i klucz

DESMAN korzysta z modeli dostępnych przez internet. Potrzebny jest darmowy klucz, który działa jak
hasło dla programu. Przy pierwszym uruchomieniu ustawienia otworzą się automatycznie.

1. W polu **Język** wybierz „Polski” lub swój język.
2. **Klucz.** Najłatwiej zacząć od jednej z tych usług. Obie są darmowe, wystarczy konto:
   - **Groq** — https://console.groq.com/keys (zaloguj się, wybierz „Create API Key” i skopiuj
     klucz).
   - **Google AI Studio** — https://aistudio.google.com/apikey (zaloguj się na konto Google,
     wybierz „Create API key” i skopiuj klucz).

   Wklej klucz w przeznaczone do tego pole. Ukryte znaki są normalne. Możesz dodać kilka kluczy
   różnych dostawców. Po wyczerpaniu limitu jednego DESMAN przechodzi na inny.
   **Nigdy nie wklejaj klucza do wiadomości**. Używaj tylko pola na klucz.
3. **Uprawnienia.** Domyślnie wszystkie są wyłączone. W teście możesz włączyć
   „korzystać z sieci poza dostawcami modeli”, aby agenci mogli czytać publiczne strony. Pozostałe
   zostaw wyłączone, chyba że właściciel poprosi inaczej.
4. **desman-global** to ostatnie pytanie. Możesz wybrać „nie teraz” i dołączyć później (rozdział
   5).

Darmowe limity są niewielkie i się zmieniają. Jeśli DESMAN informuje o wyczerpaniu limitu, praca
ruszy po jego odnowieniu, zwykle po minucie, czasem po dniu. Jeśli przez ponad godzinę nie
otrzymasz odpowiedzi, zgłoś to (rozdział 7).

## 3. Jak korzystać

Okno DESMAN to strona w przeglądarce:

- **Po lewej** są rozmowy, a na dole **Ustawienia**, **Zgłoś problem** i **Wiadomości**.
- **Pośrodku** jest rozmowa. Napisz własnymi słowami, czego potrzebujesz.
- **Po prawej** są **Zadania** (co robi DESMAN) i **Moduły** (które modele pracują).

Na proste pytanie DESMAN odpowiada od razu. Większa prośba („napisz…”, „zrób…”, „sprawdź…”) staje
się *zadaniem*. DESMAN planuje kroki, wykonuje je, sprawdza i opisuje wynik w rozmowie. Pliki są w
folderze zadania. DESMAN pokazuje, gdzie go znaleźć.

Jeśli krok się nie powiedzie, DESMAN pyta, co robić. Pod rozmową pojawią się
**Ponów z tą odpowiedzią**, **Zaakceptuj bez zmian** i **Oznacz niepowodzenie**. Możesz też
napisać, co trzeba poprawić.

Zmiany ustawień, uprawnień lub kluczy wymagają kliknięcia „Tak” w osobnym oknie. Jeśli okno pojawi
się niespodziewanie, kliknij „Nie”.

Wpisz `/` w polu rozmowy, by zobaczyć polecenia. `/help` otwiera pomoc, a `/new` zaczyna nową
rozmowę. Możesz też po prostu zapytać DESMAN, jak coś zrobić.

## 4. Co wypróbować

Używaj programu tak, jak ci wygodnie. Na początek:

1. Zapytaj o coś prostego, np. „co potrafisz?” albo „jak cię skonfigurować?”.
2. Poproś o tekst, np. „napisz list do sąsiadów o sobotnim zebraniu” albo „zaplanuj dwudniowy
   wyjazd do Krakowa”.
3. Poproś o kod, np. „napisz program w Pythonie liczący słowa w tekście”. W systemie Linux po
   włączeniu „uruchamiać polecenia powłoki” DESMAN także go uruchomi.
4. Kontynuuj zadanie, pisząc „teraz dodaj…”.
5. Zamknij okno w trakcie zadania i otwórz je ze skrótu. Zadanie powinno być kontynuowane.
6. Poproś własnymi słowami o zmianę ustawienia, np. „odpowiadaj krótko”. Powinno pojawić się okno
   potwierdzenia.
7. Dołącz do sieci (rozdział 5) i napisz do innego uczestnika (rozdział 6).

Zapamiętaj lub zanotuj wszystko, co było dziwne, niejasne lub powolne. To też są problemy.

## 5. Sieć desman-global

desman-global łączy komputery uczestników. Twój komputer może wykonać małe części cudzych zadań za
pomocą *udostępnionych* modeli. Do dużych zadań sieć łączy wolne komputery w grupy. Twoje zadania
mają pierwszeństwo. Inne są wykonywane tylko wtedy, gdy twój DESMAN jest wolny.

**Aby dołączyć:**

1. **Ustawienia** → **desman-global**.
2. W polu **Rejestr** wklej adres serwera od właściciela. Zaczyna się od `https://`.
3. Kliknij **Dołącz**. Wyświetli się odcisk twojego węzła.

**Aby udostępnić model** (opcjonalnie), wybierz go w tej samej sekcji, ustaw
**Wywołania dziennie** i **Tokeny dziennie**, a potem kliknij **Udostępnij**. Przycisk **Odbierz**
wycofuje model. Na razie można udostępnić tylko model *lokalny* (Ollama działająca na twoim
komputerze). Zasady bezpłatnych kluczy do usług w chmurze zabraniają używania ich do pracy dla
innych osób. Klucze pozostają na komputerze. Twój DESMAN sam wywołuje model i oddaje sieci tylko
wynik.

**Aby opuścić sieć**, kliknij **Opuść** w tej samej sekcji. Sieć natychmiast przestanie otrzymywać
dane z twojego komputera.

Sieć działa, gdy strona DESMAN jest otwarta. Jeśli serwer jest niedostępny, DESMAN działa
samodzielnie i ponawia próbę później.

## 6. Komunikator

Uczestnicy sieci mogą do siebie pisać. Wiadomości są szyfrowane. Operator widzi, kto, do kogo i
kiedy pisze, ale nie widzi treści.

1. **Nazwa.** Wpisz `/global name twoja_nazwa` w rozmowie. Użyj 3–32 znaków: małych liter alfabetu
   łacińskiego, cyfr, `_` i `-`. Inni zobaczą tę nazwę.
2. **Kontakt.** Możesz pisać do osób, które też się zgodziły. Wyślij prośbę przez
   `/global contact add nazwa`. Druga osoba zaakceptuje ją tym samym poleceniem z twoją nazwą.
3. **Pisanie.** Użyj `/global send nazwa "treść wiadomości"`. Tekst z kilku słów ujmij w proste
   cudzysłowy `"`. Dotyczy to też dalszych poleceń.
4. **Czytanie.** Kliknij **Wiadomości** po lewej lub użyj `/messages` albo `/messages nazwa`.
5. **Blokowanie.** Użyj `/global block nazwa`. Druga osoba nie otrzyma powiadomienia.
   `/global unblock nazwa` usuwa blokadę.
6. **Skarga.** Użyj `/global complain nazwa "co się stało"`. Aby pokazać operatorowi konkretną
   wiadomość, dodaj `--message NUMER`. Numer jest widoczny podczas czytania.
   **Treść tej wiadomości trafi do operatora**.

DESMAN nigdy nie wykonuje wiadomości od innych jako zadań. Zawarte w nich linki i pliki niczego
nie uruchamiają. Jeśli stracisz plik klucza węzła, np. po całkowitym usunięciu programu, utracisz
też wiadomości.

## 7. Zgłaszanie problemu

Do wysłania zgłoszenia potrzebne jest połączenie z siecią (rozdział 5).

1. Kliknij **Zgłoś problem** w lewym dolnym rogu.
2. Opisz, co zrobiono, jaki wynik miał się pojawić i co stało się naprawdę. Na przykład: „Po
   prośbie o list zadanie od 20 minut stoi na kroku 2 z 3”.
3. Okno pokaże **dokładną treść** dla operatora: opis oraz wersje programu i systemu. Przeczytaj
   tekst i kliknij „Tak”.

Klucze nie trafiają do zgłoszeń. Jeśli DESMAN znajdzie w tekście twój klucz, niczego nie wyśle.
Nie wysyła też rozmów, profilu ani plików. Aby dołączyć historię konkretnego zadania, wpisz
`/global report "opis" --task NUMER`. Okno pokaże dokładnie, co zostanie dodane.

Jeśli sieć lub program nie działa i nie można wysłać zgłoszenia, napisz do właściciela zwykłą
drogą i w miarę możliwości dołącz zrzut ekranu.

## 8. Aktualizacje

Aktualizacje wymagają twojej zgody. Gdy właściciel zgłosi nową wersję, połącz się z siecią
(rozdział 5). Użyj `/global update --show`, aby ją zobaczyć, i `/global update --apply NUMER`, aby
ją zainstalować. DESMAN pokaże zmiany i poprosi o potwierdzenie. Po instalacji zamknij DESMAN i
otwórz ponownie. Jeśli pojawi się komunikat o „chronionych plikach”, wpisz `/protected approve` i
potwierdź w oknie, aby przyjąć nowe zasady. Poprzednią wersję przywraca `/global update --revert`.

## 9. Usuwanie programu

- **Windows:** zamknij DESMAN, naciśnij Win+R, wklej
  `%LOCALAPPDATA%\DESMAN\app\install\install.bat -Uninstall` i naciśnij Enter.
- **Linux:** zamknij DESMAN i wykonaj
  `sh ~/.local/share/desman-app/install/install.sh --uninstall` w terminalu.

Program zostanie usunięty, a ustawienia (z kluczami) i historia zostaną na wypadek ponownej
instalacji. Aby wyczyścić wszystko, dodaj `-Purge` w Windows lub `--purge` w systemie Linux.
DESMAN pokaże, co usunie, i poprosi o ponowne potwierdzenie.

Możesz unieważnić utworzone klucze na stronach dostawców (linki w rozdziale 2).
