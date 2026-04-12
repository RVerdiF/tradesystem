"""
Testes para a Fase 4 — ML Modeling.

Valida a purga/embargo para evitar data leakage, K-fold validado no tempo, 
e logica do classificador e do bet sizing (Kelly Fracionário).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modeling.purge_embargo import apply_embargo, get_train_times, purge_and_embargo
from src.modeling.cv import PurgedKFold
from src.modeling.classifier import MetaClassifier
from src.modeling.bet_sizing import compute_kelly_fraction, discretize_bet


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_events():
    """
    Simula uma série temporal de eventos de trading.
    t0 (index) = entrada do trade
    t1 (values) = saída do trade (barreira atingida)
    """
    # 20 trades consecutivos sem sobreposição longa
    t0 = pd.date_range("2024-01-01 10:00:00", periods=20, freq="10min")
    t1 = t0 + pd.Timedelta(minutes=5)
    
    # Adiciona intencionalmente uma sobreposição severa no trade 5 
    # (fica aberto por 30 minutos, sobrepondo os trades 6 e 7)
    t1.values[5] = t0[5] + np.timedelta64(30, 'm')
    
    return pd.Series(index=t0, data=t1, name="t1")


@pytest.fixture
def synthetic_features(synthetic_events):
    """Features simples aleatórias atreladas aos eventos."""
    return pd.DataFrame(
        np.random.randn(len(synthetic_events), 3),
        index=synthetic_events.index,
        columns=["f1", "f2", "f3"]
    )


@pytest.fixture
def synthetic_labels(synthetic_events):
    """Labels binários atrelados aos eventos."""
    return pd.Series(np.random.randint(0, 2, len(synthetic_events)), index=synthetic_events.index)


# ---------------------------------------------------------------------------
# Testes — Purga e Embargo
# ---------------------------------------------------------------------------
class TestPurgeEmbargo:
    
    def test_get_train_times_removes_overlap(self, synthetic_events):
        """A purga deve remover rigorosamente apenas trades que sobrepõem o teste."""
        test_start = synthetic_events.index[6]  # 11:00:00
        test_end = synthetic_events.iloc[6]     # 11:05:00
        test_times = pd.Series(index=[test_start], data=[test_end])
        
        train_times = get_train_times(synthetic_events, test_times)
        
        # O trade 6 foi o bloco de teste, logo não deve estar no treino
        assert test_start not in train_times.index
        
        # O trade 5 tinha começo ANTES de 11:00 mas durava ATÉ 11:30!
        # Ele engloba e contamina o teste. Logo, DEVE ter sido expurgado!
        assert synthetic_events.index[5] not in train_times.index
        
        # O trade 7 começa > 11:05 (11:10), logo DEVE estra no treino.
        assert synthetic_events.index[7] in train_times.index

    def test_embargo_creates_gap(self, synthetic_events):
        """O embargo deve remover trades que começam logo após o término do teste."""
        test_start = synthetic_events.index[10] # 11:40
        test_end = synthetic_events.iloc[10]    # 11:45
        test_times = pd.Series(index=[test_start], data=[test_end])
        
        # Treino "purged" ainda tem todos os da frente
        train_times = synthetic_events.drop(test_start) 
        
        # Vamos pedir embargo de 12 minutos via step absoluto (evita calculo relativo pct pro teste unitario)
        valid = apply_embargo(train_times, test_times, step=pd.Timedelta(minutes=12))
        
        # O trade 11 (T0 = 11:50) acontece APENAS 5 minutos pós o teste (terminou 11:45).
        # Logo cai no embargo de 12min. Deve estar de fora.
        assert synthetic_events.index[11] not in valid.index
        
        # O trade 12 (T0 = 12:00) acontece 15 minutos apos teste. Fica no Treino.
        assert synthetic_events.index[12] in valid.index


# ---------------------------------------------------------------------------
# Testes — Purged K-Fold CV
# ---------------------------------------------------------------------------
class TestPurgedKFold:
    
    def test_cv_splits_correctly(self, synthetic_events, synthetic_features):
        """Deve particionar os folds sempre respeitando exclusão de índices expurgados."""
        kf = PurgedKFold(samples_info=synthetic_events, n_splits=3, pct_embargo=0.0)
        
        splits = list(kf.split(synthetic_features))
        assert len(splits) == 3
        
        for i, (train_idx, test_idx) in enumerate(splits):
            # Não pode haver interseção entre índices numéricos de treino e teste
            assert len(np.intersect1d(train_idx, test_idx)) == 0
            
            # Testa se sobrou treino (garante q as particoes nao sumiram com a amostra inteira)
            assert len(train_idx) > 0
            assert len(test_idx) > 0


# ---------------------------------------------------------------------------
# Testes — Classificador Secundário
# ---------------------------------------------------------------------------
class TestMetaClassifier:
    
    def test_classifier_fit_predict(self, synthetic_features, synthetic_labels):
        """Garante contrato do SciKit-Learn."""
        clf = MetaClassifier(n_estimators=10, max_depth=2)
        
        assert not clf.is_fitted_
        clf.fit(synthetic_features, synthetic_labels)
        assert clf.is_fitted_
        
        preds = clf.predict(synthetic_features)
        assert len(preds) == len(synthetic_labels)
        assert set(preds).issubset({0, 1})
        
        probs = clf.predict_proba(synthetic_features)
        # Probabilidades shape (n_samples, n_classes)
        assert probs.shape == (len(synthetic_features), 2)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

    def test_feature_importances(self, synthetic_features, synthetic_labels):
        clf = MetaClassifier(n_estimators=10, max_depth=2).fit(synthetic_features, synthetic_labels)
        imp = clf.feature_importances(feature_names=["f1", "f2", "f3"])
        
        assert len(imp) == 3
        assert list(imp.index).count("f1") > 0  # Garante q mapeou string names
        assert round(imp.sum(), 3) == 1.0


# ---------------------------------------------------------------------------
# Testes — Bet Sizing
# ---------------------------------------------------------------------------
class TestBetSizing:
    
    def test_kelly_logic(self):
        """Valida que o tamanho otimo do Kelly atende matematicamente a equacao."""
        # Se tenho 60% chance de ganhar (e pago/lucro 1:1) => p=0.6, q=0.4, odds=1
        # Kelly = p - q/odds = 0.6 - 0.4/1 = 0.20 (20% banca)
        bet = compute_kelly_fraction(prob_win=0.6, odds=1.0, fraction=1.0) # Kelly Puro
        assert np.isclose(bet, 0.20)
        
        # Kelly Fracionário (ex half-kelly = 0.5) => aposta 10%
        bet_frac = compute_kelly_fraction(prob_win=0.6, odds=1.0, fraction=0.5) 
        assert np.isclose(bet_frac, 0.10)
        
    def test_kelly_clips_negative_edge(self):
        """Probabilidades ruins devem gerar zeros (sem short ou alavancagem suicidada)."""
        # 40% chance vitoria 1:1 = Kelly -0.2 => zeroed
        bet = compute_kelly_fraction(prob_win=0.4, odds=1.0, fraction=1.0)
        assert bet == 0.0
        
    def test_kelly_clips_extreme_prob(self):
        """Kelly não pode ser > 100% da banca permitida para a posição mesmo se 99% win"""
        bet = compute_kelly_fraction(prob_win=0.99, odds=1.0, fraction=2.0)
        assert bet == 1.0
        
    def test_discretize_position(self):
        """Deve converter o escalar fracionário em contratos redondos, até o max permitido"""
        p_series = pd.Series([0.2, 0.8, 0.5])
        
        # Com cap = 5 lotes totais:  
        # Kelly 20% = 1 lote, Kelly 80% = 4 lotes, Kelly 50% = 2.5 (arredondado pra 2 dependendo np.round half-to-even)
        lotes = discretize_bet(p_series, max_position=5, step_size=1)
        assert lotes.iloc[0] == 1
        assert lotes.iloc[1] == 4
        # np.round(2.5) normal é 2. Porém np bankar round half-even pode dar 2
        assert lotes.iloc[2] in [2, 3] 

        # Deve obedecer ao Max absoluto configurado (ex: max=3 lotes)
        lotes_capped = discretize_bet(p_series, max_position=3, step_size=1)
        # O 80% que daria ~2.4 lotes arredonda para 2.
        assert lotes_capped.iloc[1] == 2
        
        # Agora sim: 100% Kelly mas max_pos = 3 => tem q ser <= 3
        p_max = pd.Series([1.0])
        lotes_cap = discretize_bet(p_max, max_position=3)
        assert lotes_cap.iloc[0] == 3


# ---------------------------------------------------------------------------
# Testes — Conviction Threshold
# ---------------------------------------------------------------------------
class TestConvictionThreshold:
    """Testa o filtro de probabilidade pre-Kelly."""

    def test_scalar_above_threshold_passes(self):
        """Probabilidade acima do threshold não é modificada."""
        from src.modeling.bet_sizing import apply_conviction_threshold
        result = apply_conviction_threshold(0.65, threshold=0.5)
        assert result == 0.65

    def test_scalar_below_threshold_zeroed(self):
        """Probabilidade abaixo do threshold é zerada."""
        from src.modeling.bet_sizing import apply_conviction_threshold
        result = apply_conviction_threshold(0.45, threshold=0.5)
        assert result == 0.0

    def test_scalar_exactly_at_threshold_passes(self):
        """Probabilidade exatamente igual ao threshold NÃO é zerada (>= semantics)."""
        from src.modeling.bet_sizing import apply_conviction_threshold
        result = apply_conviction_threshold(0.5, threshold=0.5)
        assert result == 0.5

    def test_series_mixed_values(self):
        """Series com valores acima e abaixo do threshold — apenas os baixos são zerados."""
        from src.modeling.bet_sizing import apply_conviction_threshold
        probs = pd.Series([0.3, 0.55, 0.70, 0.49, 0.60])
        filtered = apply_conviction_threshold(probs, threshold=0.5)
        assert filtered.iloc[0] == 0.0   # 0.3 < 0.5 → zerado
        assert filtered.iloc[1] == 0.55  # passa
        assert filtered.iloc[2] == 0.70  # passa
        assert filtered.iloc[3] == 0.0   # 0.49 < 0.5 → zerado
        assert filtered.iloc[4] == 0.60  # passa

    def test_threshold_to_kelly_chain_produces_zero(self):
        """Probabilidade abaixo do threshold → kelly final deve ser zero."""
        from src.modeling.bet_sizing import apply_conviction_threshold, compute_kelly_fraction
        prob_filtered = apply_conviction_threshold(0.40, threshold=0.5)
        kelly = compute_kelly_fraction(prob_win=prob_filtered, odds=1.0, fraction=1.0)
        # prob=0 → f* = 0 - 1/1 = -1 → clipped to 0
        assert kelly == 0.0

    def test_does_not_mutate_input_series(self):
        """A função não deve alterar a Series original (imutabilidade)."""
        from src.modeling.bet_sizing import apply_conviction_threshold
        original = pd.Series([0.3, 0.7])
        original_copy = original.copy()
        apply_conviction_threshold(original, threshold=0.5)
        pd.testing.assert_series_equal(original, original_copy)
