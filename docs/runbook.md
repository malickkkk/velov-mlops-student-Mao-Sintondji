# Runbook : API velov-availability

> Template à compléter au fil des sessions. Public cible : une personne d'astreinte
> qui ne connaît pas le projet et doit rétablir le service à 3 h du matin.

## 1. Vue d'ensemble
- Ce que fait le service, en deux phrases : l'API prédit le nombre de vélos disponibles
  dans une heure pour une station Vélo'v. Chaque prédiction peut être persistée dans
  PostgreSQL avec les entrées, la sortie, la version du modèle et les timestamps UTC.
- Schéma d'architecture : `client -> velov-api:8000 -> PostgreSQL:5432`.
- Dépendances : image Docker de l'API, modèle généré dans `/app/models`, PostgreSQL
  et le volume Docker `pgdata`.

## 2. Démarrer / arrêter
```bash
# depuis la racine du dépôt
MODEL_VERSION=1.0.0 \
GIT_COMMIT="$(git rev-parse --short HEAD)" \
docker compose up --build -d

# arrêter les conteneurs en conservant les données
docker compose down

# redémarrer uniquement l'API
docker compose restart api
```

## 3. Vérifier que tout va bien
| Vérification | Commande | Résultat attendu |
|---|---|---|
| Le process répond | `curl -s http://<hôte>:8000/health` | `{"status":"ok"}` |
| Le modèle est chargé | `curl -s http://<hôte>:8000/ready` | HTTP 200 + version |
| Une prédiction fonctionne | `curl -X POST http://<hôte>:8000/v1/predict -H "Content-Type: application/json" -d '{"station_id":3,"timestamp":"2026-10-06T08:00:00+02:00","capacity":20,"bikes_available":12,"temperature":14.5,"is_raining":false}'` | HTTP 200 + `model_version` |
| La prédiction est en base | `docker compose exec db psql -U velov -d velov -c "SELECT COUNT(*) FROM predictions;"` | compteur supérieur à zéro après un appel |

Les logs sont consultables avec `docker compose logs -f api` et
`docker compose logs -f db`.

## 4. Incidents connus
| Symptôme | Cause probable | Diagnostic | Action |
|---|---|---|---|
| `/ready` renvoie 503 | Modèle absent ou empreinte invalide | `docker compose logs api` | Reconstruire l'image avec `docker compose build --no-cache` |
| Le conteneur redémarre en boucle | Erreur de démarrage ou healthcheck en échec | `docker compose ps` puis `docker compose logs api` | Corriger l'erreur affichée puis reconstruire |
| Latence élevée | Base indisponible ou charge importante | `docker compose logs api db` | Vérifier `docker compose ps` et la santé PostgreSQL |

## 5. Déployer une nouvelle version du modèle
1. Construire et démarrer avec une nouvelle version et le SHA du commit :
   `MODEL_VERSION=1.1.0 GIT_COMMIT="$(git rev-parse --short HEAD)" docker compose up --build -d`.
2. Vérifier `/health`, `/ready`, `/v1/predict` et la table `predictions`.
3. Retour arrière (rollback) : relancer avec l'ancienne `MODEL_VERSION`, le commit
   correspondant, puis reconstruire l'image.

## 6. Contacts et escalade
- Consulter les logs de l'API et de PostgreSQL avant toute intervention.
- Conserver le volume `pgdata` lors d'un arrêt normal ; ne pas utiliser
  `docker compose down -v` sauf si la suppression des données est volontaire.
