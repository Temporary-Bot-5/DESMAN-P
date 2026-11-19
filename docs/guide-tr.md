# DESMAN(P): test katılımcıları için kılavuz

Yardım ettiğiniz için teşekkürler. Bu kılavuz DESMAN'ı nasıl kuracağınızı, neleri
deneyebileceğinizi ve sorunları nasıl bildireceğinizi anlatır. Önce 1–3. bölümleri okuyup kalanına
gerektiğinde bakabilirsiniz.

## 0. Önce bunu okuyun

- **Bu bir prototip.** Program çökebilir, takılabilir, tuhaf yanıtlar verebilir ve bitmemiş işleri
  kaybedebilir. Önemli işlerinizi emanet etmeyin ve tekrarlanamayacak görevler vermeyin.
  Bulduğunuz sorunlar testin en önemli sonucudur. Lütfen hepsini bildirin (7. bölüm).
- **Anahtarlarınız sizde kalır.** Model erişim anahtarı (API anahtarı) yalnızca bilgisayarınızdaki
  bir dosyada tutulur ve sadece siz görebilirsiniz. DESMAN ağına, sahibine, modele veya sohbet
  geçmişine gitmez. Ağa yalnızca anahtarın *parmak izi* gider. Bu kısa karakter dizisi anahtarları
  birbirinden ayırır ama asıl anahtarı elde etmeyi sağlamaz.
- **Ağı yalnız siz açarsınız.** **Katıl** düğmesine basana kadar (5. bölüm) DESMAN, sahibinin
  sunucusuyla hiç bağlantı kurmaz. Görevleriniz, sohbetleriniz ve dosyalarınız hep
  bilgisayarınızda kalır. İstekler yalnızca anahtarını eklediğiniz modellerin sağlayıcılarına
  gider.
- **Ağa katıldığınızda operatörün (sunucu sahibi) görebildikleri:**
  - düğümünüzün anahtar parmak izi, son bağlantı zamanı, ödünç verdiğiniz modeller ve günlük
    kotaları (sağlayıcı, model ve anahtar parmak izi; anahtarın kendisi görünmez);
  - bilgisayarınızın ağ için yaptığı iş ve sonucu;
  - gönderdiğiniz sorun bildirimleri, onay penceresindeki metinle aynı şekilde (7. bölüm);
  - mesajlaşmada adınız ve kimin kime ne zaman yazdığı. Mesajlar şifrelidir ve operatör içeriği
    okuyamaz. **Şikâyet durumunda** bildirdiğiniz mesajın metni operatöre gider;
  - her web sitesinde olduğu gibi, bağlantının geldiği IP adresi.
- **Windows'ta kod çalıştırılmaz.** DESMAN'ın ajanları programları yalnızca Linux'ta bulunan,
  “korumalı alan” denen yalıtılmış ortamda çalıştırır. Windows'ta DESMAN kod yazar ve kodu
  çalıştırarak kontrol etmediğini belirtir.

## 1. Kurulum

Sahip, kurulum düğmelerinin olduğu sayfanın bağlantısını gönderir. İnternet bağlantısı gerekir.
Yönetici yetkisine gerek yoktur.

### Windows

1. Sayfada **Windows** düğmesine basın. `install.bat` indirilir. Tarayıcı dosyanın tutulup
   tutulmayacağını sorarsa “Sakla”yı seçin.
2. İndirilenler klasörünü açın ve `install.bat` dosyasına çift tıklayın. Mavi “Windows
   bilgisayarınızı korudu” penceresi çıkarsa “Ek bilgi” → “Yine de çalıştır” seçeneğine tıklayın.
3. Siyah bir pencere açılır. Yükleyici gerekli yazılımları (uv, Python ve eksikse Git) kurmak için
   izin ister. `y` yazıp Enter'a basın. Bu birkaç dakika sürer.
4. Sonunda tarayıcı DESMAN sayfasını açar ve masaüstünde **DESMAN** kısayolu görünür. Bundan sonra
   programı bu kısayoldan açın.

### Linux

1. Sayfada **Linux** düğmesine basın ve gösterilen komutu kopyalayın.
2. Terminali açın, komutu Ctrl+Shift+V ile yapıştırın ve Enter'a basın.
3. Git yoksa yükleyici kurulum komutunu gösterir. Komutu çalıştırıp 2. adımı yineleyin.
4. Sorulara `y` ile yanıt verin. Sonunda tarayıcı açılır ve **DESMAN** uygulama menüsünde ve
   masaüstünde görünür. Masaüstü kısayolu çalıştırmak için izin isterse izin verin.

Ayrıntılar için [install.md](install.md) dosyasına bakın.

## 2. İlk açılış ve anahtar

DESMAN internet üzerinden modellere erişir. Bunun için program parolası gibi çalışan ücretsiz bir
anahtar gerekir. İlk açılışta ayarlar kendiliğinden açılır.

1. **Dil** bölümünde “Türkçe”yi veya istediğiniz dili seçin.
2. **Anahtar.** Başlamak için aşağıdaki hizmetlerden birini seçmek en kolay yoldur. İkisi de
   ücretsizdir ve yalnızca hesap gerekir:
   - **Groq** — https://console.groq.com/keys (oturum açın, “Create API Key”i seçin ve anahtarı
     kopyalayın).
   - **Google AI Studio** — https://aistudio.google.com/apikey (Google hesabınızla oturum açın,
     “Create API key”i seçin ve anahtarı kopyalayın).

   Anahtarı ayarlardaki ilgili alana yapıştırın. Karakterlerin görünmemesi normaldir. Farklı
   sağlayıcılardan birden çok anahtar ekleyebilirsiniz. Birinin kotası dolduğunda DESMAN diğerine
   geçer. **Anahtarı asla sohbet mesajına yapıştırmayın**. Yalnızca anahtar alanını kullanın.
3. **İzinler.** Başlangıçta hepsi kapalıdır. Test için “model sağlayıcıları dışında ağı kullanmak”
   iznini açarak ajanların herkese açık web sayfalarını okumasını sağlayabilirsiniz. Sahip aksini
   istemedikçe diğerlerini kapalı tutun.
4. **desman-global**, kurulumun son sorusudur. “Şimdi değil” deyip daha sonra katılabilirsiniz (5.
   bölüm).

Ücretsiz kotalar düşüktür ve değişebilir. DESMAN kota doldu derse kota yenilendiğinde iş sürer. Bu
genellikle bir dakika, bazen bir gün alır. Bir saatten uzun süre yanıt gelmezse bildirin (7.
bölüm).

## 3. Kullanım

DESMAN penceresi tarayıcıda bir sayfa olarak açılır:

- **Solda** sohbetler, alt kısımda **Ayarlar**, **Sorun bildir** ve **Mesajlar** düğmeleri vardır.
- **Ortada** sohbet alanı vardır. İhtiyacınızı kendi sözlerinizle yazın.
- **Sağda** **Görevler** (DESMAN'ın yaptığı işler) ve **Modüller** (çalışan modeller) görünür.

DESMAN basit soruları hemen yanıtlar. “Yaz…”, “yap…”, “araştır…” gibi daha kapsamlı istekler birer
*görev* olur. DESMAN adımları planlar, uygular, kontrol eder ve sonucu sohbette anlatır. Görev
dosyaları ayrı bir klasörde tutulur. DESMAN klasörün yerini gösterir.

Bir adım başarısız olursa DESMAN ne yapılacağını sorar. Sohbetin altında
**Bu yanıtla yeniden dene**, **Olduğu gibi kabul et** ve **Başarısız olarak işaretle** düğmeleri
çıkar. Neyin düzeltilmesi gerektiğini yazarak da yanıt verebilirsiniz.

Ayarlar, izinler veya anahtarlar ancak ayrı bir pencerede “Evet” düğmesine basınca değişir.
Pencere beklemediğiniz bir anda açılırsa “Hayır” düğmesine basın.

Komutları görmek için sohbet alanına `/` yazın. `/help` yardımı açar, `/new` yeni sohbet başlatır.
Bir şeyin nasıl yapıldığını DESMAN'a doğrudan da sorabilirsiniz.

## 4. Neleri deneyebilirsiniz?

Programı istediğiniz gibi kullanın. Başlangıç için:

1. “Neler yapabilirsin?” veya “Seni nasıl kurarım?” gibi basit bir soru sorun.
2. “Cumartesi toplantısı hakkında komşularıma bir mektup yaz” veya “İzmir'e iki günlük bir gezi
   planla” gibi bir yazı görevi verin.
3. “Bir metindeki kelimeleri sayan Python programı yaz” gibi bir kod görevi verin. Linux'ta
   “kabuk komutları çalıştırmak” izni açıksa DESMAN programı da çalıştırır.
4. “Şimdi buna … ekle” diyerek bir görevi sürdürün.
5. Görev devam ederken pencereyi kapatıp kısayolla tekrar açın. Görev devam etmeli.
6. “Kısa yanıt ver” gibi bir ayar değişikliği isteyin. Bir onay penceresi açılmalı.
7. Ağa katılın (5. bölüm) ve başka bir katılımcıya yazın (6. bölüm).

Tuhaf, anlaşılmaz veya yavaş gelen şeyleri hatırlayın ya da not alın. Bunlar da sorun sayılır.

## 5. desman-global ağı

desman-global katılımcıların bilgisayarlarını birbirine bağlar. Bilgisayarınız *ödünç verdiğiniz*
modellerle başkalarının görevlerinin küçük bölümlerini yapabilir. Ağ, büyük görevler için boş
bilgisayarları gruplar hâlinde toplar. Sizin görevleriniz her zaman önceliklidir. Başkalarının
işleri yalnızca DESMAN'ınız boşta olduğunda çalışır.

**Katılmak için:**

1. **Ayarlar** → **desman-global** bölümünü açın.
2. **Kayıt sunucusu** alanına sahibin gönderdiği sunucu adresini yapıştırın. Adres `https://` ile
   başlar.
3. **Katıl** düğmesine basın. Düğümünüzün parmak izi görünür.

**Model ödünç vermek için** (isteğe bağlı) aynı bölümde modeli seçin, **Günlük çağrı** ve
**Günlük token** değerlerini ayarlayıp **Ödünç ver** düğmesine basın. **Geri al** düğmesi modeli
geri alır. Şimdilik yalnızca bilgisayarınızda çalışan *yerel* modeller (Ollama) ödünç verilebilir.
Ücretsiz bulut anahtarlarının koşulları başkaları için çalışmaya izin vermez. Anahtarlar
bilgisayarınızda kalır. DESMAN'ınız çağrıyı yapar ve ağa yalnızca sonucu yollar.

**Ayrılmak için** aynı bölümde **Ayrıl** düğmesine basın. Ağ bilgisayarınızdan veri almayı hemen
bırakır.

Ağ, DESMAN sayfası açıkken çalışır. Sunucuya ulaşılamazsa DESMAN tek başına çalışmaya devam eder
ve daha sonra yeniden dener.

## 6. Mesajlaşma

Ağ katılımcıları birbirine yazabilir. Mesajlar şifrelenir. Operatör kimin kime ne zaman yazdığını
görür ancak içeriği okuyamaz.

1. **Ad.** Sohbete `/global name adınız` yazın. 3–32 karakter kullanın: küçük Latin harfleri,
   rakamlar, `_` ve `-`. Bu adı diğer katılımcılar görür.
2. **Kişi ekleme.** Yalnızca size yazmaya onay veren kişilere yazabilirsiniz.
   `/global contact add ad` ile istek gönderin. Karşı taraf aynı komutu sizin adınızla kullanarak
   kabul eder.
3. **Yazma.** `/global send ad "mesaj metni"` komutunu kullanın. Birden fazla sözcük varsa metni
   düz çift tırnak `"` içine alın. Aşağıdaki komutlar için de bu geçerlidir.
4. **Okuma.** Soldaki **Mesajlar** düğmesini, `/messages` veya `/messages ad` komutunu kullanın.
5. **Engelleme.** `/global block ad` kullanın. Karşı tarafa bildirim gitmez. `/global unblock ad`
   engeli kaldırır.
6. **Şikâyet.** `/global complain ad "ne oldu"` komutunu kullanın. Operatöre belirli bir mesajı
   göstermek için `--message NUMARA` ekleyin. Numara mesajları okurken görünür.
   **O mesajın metni operatöre gider**.

DESMAN başkalarından gelen mesajları görev olarak çalıştırmaz. Mesajlardaki bağlantılar ve
dosyalar bir işlem başlatmaz. Programı tamamen kaldırmak gibi bir nedenle düğümün anahtar
dosyasını kaybederseniz mesajları da kaybedersiniz.

## 7. Sorun bildirme

Bildirim göndermek için ağa bağlı olmalısınız (5. bölüm).

1. Sol alttaki **Sorun bildir** düğmesine basın.
2. Ne yaptığınızı, ne beklediğinizi ve ne olduğunu yazın. Örneğin: “Mektup yazmasını istedim.
   Görev üç adımın ikincisinde 20 dakikadır takılı kaldı.”
3. Açılan pencerede operatöre gidecek **metnin tamamını** görürsünüz. Açıklamanızla birlikte
   program ve sistem sürümleri yer alır. Okuyup “Evet” düğmesine basın.

Anahtarlar bildirime eklenmez. DESMAN metinde anahtarınızı bulursa hiçbir şey göndermez.
Sohbetler, profil ve görev dosyaları da gönderilmez. Belirli bir görevin geçmişini eklemek için
`/global report "açıklama" --task NUMARA` yazın. Pencere tam olarak nelerin ekleneceğini gösterir.

Ağa bağlanamadığınız veya program açılmadığı için gönderemiyorsanız, sahibine her zamanki yoldan
yazın. Mümkünse ekran görüntüsü ekleyin.

## 8. Güncellemeler

Güncellemeler kendiliğinden kurulmaz. Sahip yeni sürümü duyurunca ağa bağlanın (5. bölüm). Sürümü
görmek için `/global update --show`, kurmak için `/global update --apply NUMARA` kullanın. DESMAN
değişiklikleri gösterip onay ister. Kurulumdan sonra programı kapatıp yeniden açın. “Korunan
dosyalar” bildirimi çıkarsa `/protected approve` yazıp pencerede onaylayarak yeni kuralları kabul
edin. Önceki sürüme dönmek için `/global update --revert` kullanın.

## 9. Kaldırma

- **Windows:** DESMAN'ı kapatın, Win+R tuşlarına basın,
  `%LOCALAPPDATA%\DESMAN\app\install\install.bat -Uninstall` komutunu yapıştırın ve Enter'a basın.
- **Linux:** DESMAN'ı kapatın ve terminalde
  `sh ~/.local/share/desman-app/install/install.sh --uninstall` komutunu çalıştırın.

Program kaldırılır. Yeniden kurmanız gerekirse diye ayarlar (anahtarlarla birlikte) ve geçmiş
kalır. Hepsini silmek için Windows'ta `-Purge`, Linux'ta `--purge` ekleyin. DESMAN neleri
sileceğini gösterir ve yeniden onay ister.

Oluşturduğunuz anahtarları sağlayıcıların sitelerinden iptal edebilirsiniz (bağlantılar 2.
bölümde).
